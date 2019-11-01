/* 
 * File:   HigherOrderTri2DSCS.C
 * Author: Raphael Lindegger
 * 
 * Created on October 20, 2019, 10:43 AM
 */

#include <master_element/HigherOrderTri2DSCS.h>

#include <master_element/MasterElementFunctions.h>
#include <master_element/HigherOrderMasterElementFunctions.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/TensorOps.h>

#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>

#include <NaluEnv.h>
#include <master_element/MasterElement.h>
#include <FORTRAN_Proto.h>

#include <BuildTemplates.h>

#include <stk_util/util/ReportHandler.hpp>

#include <array>
#include <vector>
#include <limits>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <element_promotion/ElementDescription.h>

namespace sierra{
namespace nalu{

KOKKOS_FUNCTION
HigherOrderTri2DSCS::HigherOrderTri2DSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature))
#ifndef KOKKOS_ENABLE_CUDA
  , nodeMap(make_node_map_tri(basis.order())),
  faceNodeMap(make_face_node_map_tri(basis.order())),
  sideNodeOrdinals_(make_side_node_ordinal_map_tri(basis.order()))
#endif
  , nodes1D_(basis.order()+1)
{
  MasterElement::nDim_ = 2;
  nodesPerElement_ = 0.5*(nodes1D_*(nodes1D_+1));

#ifndef KOKKOS_ENABLE_CUDA
  // set up integration rule and relevant maps for scs
  set_interior_info();

  // set up integration rule and relevant maps for faces
  set_boundary_info();

  // temporary, copied from quad, does not work with tri but currently unused
  expFaceShapeDerivs_ = basis_.eval_deriv_weights(intgExpFace_);
#endif
}

double parametric_distance_tri(const double* x) // TODO check purpose and adapt to tri
{
  double absXi  = std::abs(x[0]);
  double absEta = std::abs(x[1]);
  return (absXi > absEta) ? absXi : absEta;
}

// Used for non-conformal and overset
// TODO adapt to TrI (copied from quad)
double HigherOrderTri2DSCS::isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord)
{
  std::array<double, 2> initialGuess = {{ 0.0, 0.0 }};
  int maxIter = 50;
  double tolerance = 1.0e-16;
  double deltaLimit = 1.0e4;

  bool converged = isoparameteric_coordinates_for_point_2d(
      basis_,
      elemNodalCoord,
      pointCoord,
      isoParCoord,
      initialGuess,
      maxIter,
      tolerance,
      deltaLimit
  );
  ThrowAssertMsg(parametric_distance_quad(isoParCoord) < 1.0 + 1.0e-6 || !converged,
      "Inconsistency in parametric distance calculation");

  return (converged) ? parametric_distance_tri(isoParCoord) : std::numeric_limits<double>::max();
}

// Not sure about its use
// TODO adapt to TrI (copied from quad)
void HigherOrderTri2DSCS::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result)
{
  const auto& weights = basis_.point_interpolation_weights(isoParCoord);
  for (int n = 0; n < nComp; ++n) {
    result[n] = ddot(weights.data(), field + n * nodesPerElement_, nodesPerElement_);
  }
}

// Set integration locations in the isoparametric coordinate frame and
// the left/right node ordinal to each integration point
void
HigherOrderTri2DSCS::set_interior_info()
{
  const int polyOrder = nodes1D_ - 1;
  const int numQuad = quadrature_.num_quad();
  auto desc = ElementDescription::create(2, polyOrder, stk::topology::TRI_3_2D);
  
  int IPCount = 0;
  int subfaceCount = 0;
  std::vector<ordinal_type> subTriNodeOrdinals(3);
  std::vector<ordinal_type> neighborSubTriNodeOrdinals(3);
  std::vector<double> subTriCentroid(2);
  std::vector<double> endLoc(2, 0.0);

  // lambda to compute the centroid of a triangle
  auto getCentroid = [&](std::vector<ordinal_type>& nodeOrdinals) {
    const double oneThird = 1.0/3.0;
    std::vector<double>centroid(2, 0.0);
    for (auto nodeOrdinal : nodeOrdinals) {
      for (int i = 0; i < 2; ++i) {        
        const double coord = desc->nodeLocs[nodeOrdinal][i];
        centroid[i] += oneThird * coord;
      }
    }
    return centroid;
  };

  // lambda to compute the integration location and weight
  auto writeIPInfo = [&](std::vector<double>& B, std::vector<double>& P, int& count, int& subFC, ordinal_type left, ordinal_type right, int orientation) {
    
    // Save BP vector (subface) for later usage in areav computation
    // Note that the coords are in the isoparametric coord frame
    intSubfaces_(subFC, 0, 0) = B[0];
    intSubfaces_(subFC, 0, 1) = B[1];
    intSubfaces_(subFC, 1, 0) = P[0];
    intSubfaces_(subFC, 1, 1) = P[1];
    subFC++;
    
    for (int j = 0; j < numQuad; ++j) {
      const double abscissa = quadrature_.abscissa(j);
      const double length = std::sqrt(pow(P[0]-B[0], 2) + pow(P[1]-B[1], 2));
      ipWeights_(count) = orientation * quadrature_.weights(j) * length;
      std::cout << "ipWeights_(" << count << ") = " << ipWeights_(count) << std::endl;
      lrscv_(2*count) = left;
      lrscv_(2*count+1) = right;
      std::cout << "left node: " << lrscv_(2*count) << ", right node: " << lrscv_(2*count+1) << std::endl;
      for (int i = 0; i < 2; ++i) {
        intgLoc_(count, i) = B[i] + 0.5 * ( abscissa + 1) * (P[i] - B[i] );
        std::cout << "intgLoc_(" << count << ", " << i << ") = " << intgLoc_(count, i) << std::endl;
      }
      count++;
    }
  };

  // Hardcode left/right node mapping and integration locations
  if (polyOrder == 1) {
    numIntPoints_ = 3;
    int orientation;

    // define L/R mappings
    lrscv_ = Kokkos::View<int*>("lsrcv_", 2*numIntPoints_);

    // standard integration location
    intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
    ipWeights_ = Kokkos::View<double*>("ipWeights_", numIntPoints_);
    intSubfaces_ = Kokkos::View<double*[2][2]>("intSubfaces_", numIntPoints_/quadrature_.num_quad());
    
    // compute subtriangle centroid
    subTriNodeOrdinals = {0, 1, 2};
    subTriCentroid = getCentroid(subTriNodeOrdinals);
    
    // bottom edge face endloc
    endLoc[0] = quadrature_.scs_end_loc(1);
    endLoc[1] = 0.0;
    orientation = 1;
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[0], subTriNodeOrdinals[1], orientation);
    
    // right edge face endloc
    endLoc[0] = quadrature_.scs_end_loc(1);
    endLoc[1] = quadrature_.scs_end_loc(1);
    orientation = 1;
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[1], subTriNodeOrdinals[2], orientation);
    
    // left edge face endloc
    endLoc[0] = 0.0;
    endLoc[1] = quadrature_.scs_end_loc(1);
    orientation = 1;
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[0], subTriNodeOrdinals[2], orientation);
  }
  else if (polyOrder == 2) {
    numIntPoints_ = 18;
    int orientation = 1;

    // define L/R mappings
    lrscv_ = Kokkos::View<int*>("lsrcv_", 2*numIntPoints_);

    // standard integration location
    intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
    ipWeights_ = Kokkos::View<double*>("ipWeights_", numIntPoints_);
    intSubfaces_ = Kokkos::View<double*[2][2]>("intSubfaces_", numIntPoints_/quadrature_.num_quad());
    
    //----------------------------------------------------------------
    // bottom left subtriangle
    
    // compute subtriangle centroid
    subTriNodeOrdinals = {0, 3, 5};
    subTriCentroid = getCentroid(subTriNodeOrdinals);
    
    // bottom edge face endloc
    endLoc[0] = quadrature_.scs_end_loc(1);
    endLoc[1] = 0.0;
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[0], subTriNodeOrdinals[1], orientation);
    
    // right neighbor subtriangle
    neighborSubTriNodeOrdinals = {3, 4, 5};
    endLoc = getCentroid(neighborSubTriNodeOrdinals);
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[1], subTriNodeOrdinals[2], orientation);
    
    // left edge face endloc
    endLoc[0] = 0.0;
    endLoc[1] = quadrature_.scs_end_loc(1);
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[0], subTriNodeOrdinals[2], orientation);
    
    //----------------------------------------------------------------
    // bottom right subtriangle
    
    // compute subtriangle centroid
    subTriNodeOrdinals = {3, 1, 4};
    subTriCentroid = getCentroid(subTriNodeOrdinals);
    
    // bottom edge face endloc
    endLoc[0] = quadrature_.scs_end_loc(2);
    endLoc[1] = 0.0;
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[1], subTriNodeOrdinals[0], orientation);
    
    // right edge face endloc
    endLoc[0] = quadrature_.scs_end_loc(2);
    endLoc[1] = quadrature_.scs_end_loc(1);
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[1], subTriNodeOrdinals[2], orientation);
    
    // left neighbor subtriangle
    neighborSubTriNodeOrdinals = {3, 4, 5};
    endLoc = getCentroid(neighborSubTriNodeOrdinals);
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[0], subTriNodeOrdinals[2], orientation);
    
    //----------------------------------------------------------------
    // top subtriangle
    
    // compute subtriangle centroid
    subTriNodeOrdinals = {5, 4, 2};
    subTriCentroid = getCentroid(subTriNodeOrdinals);
    
    // bottom neighbor subtriangle
    neighborSubTriNodeOrdinals = {3, 4, 5};
    endLoc = getCentroid(neighborSubTriNodeOrdinals);
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[1], subTriNodeOrdinals[0], orientation);
    
    // right edge face endloc
    endLoc[0] = quadrature_.scs_end_loc(1);
    endLoc[1] = quadrature_.scs_end_loc(2);
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[2], subTriNodeOrdinals[1], orientation);
    
    // left edge face endloc
    endLoc[0] = 0.0;
    endLoc[1] = quadrature_.scs_end_loc(2);
    writeIPInfo(subTriCentroid, endLoc, IPCount, subfaceCount, subTriNodeOrdinals[2], subTriNodeOrdinals[0], orientation);
  }
  else {
    ThrowErrorMsg("Only P1 and P2 is defined for TRI_2_2D elements.");
  }
}

void
HigherOrderTri2DSCS::set_boundary_info()
{
  const int numFaces = 2*nDim_;
  const int nodesPerFace = nodes1D_;
  ipsPerFace_ = nodesPerFace*quadrature_.num_quad();

  const int numFaceIps = numFaces*ipsPerFace_;

  oppFace_ =Kokkos::View<int*>("oppFace_", numFaceIps);
  ipNodeMap_ = Kokkos::View<int*>("ipNodeMap_", numFaceIps);
  oppNode_ = Kokkos::View<int*>("oppNode", numFaceIps);
  intgExpFace_=Kokkos::View<double**>("intgExpFace_", numFaceIps,nDim_);


  auto face_node_number = [&] (int number,int faceOrdinal)
  {
    return faceNodeMap(faceOrdinal,number);
  };

  int scalar_index = 0;
  int faceOrdinal = 0; //bottom face
  int oppFaceIndex = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = nodeMap(k, 1);
    std::cout << "nearNode: " << nearNode << std::endl;
    std::cout << "oppNode: " << oppNode << std::endl;

    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
//      oppFace_(scalar_index) = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_; // Not sure what this is for
      
      intgExpFace_(scalar_index, 0) = quadrature_.integration_point_location(k,j);
      std::cout << "intgExpFace_(" << scalar_index << ", " << 0 << ") = " << intgExpFace_(scalar_index, 0) << std::endl;
      intgExpFace_(scalar_index, 1) = 0.0;
      std::cout << "intgExpFace_(" << scalar_index << ", " << 1 << ") = " << intgExpFace_(scalar_index, 1) << std::endl;

      ++scalar_index;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 1; //right face
  oppFaceIndex = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = nodeMap(k, nodes1D_-2);
    std::cout << "nearNode: " << nearNode << std::endl;
    std::cout << "oppNode: " << oppNode << std::endl;

    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
//      oppFace_(scalar_index) = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_; // Not sure what this is for

      intgExpFace_(scalar_index, 0) = quadrature_.integration_point_location(k,j);
      std::cout << "intgExpFace_(" << scalar_index << ", " << 0 << ") = " << intgExpFace_(scalar_index, 0) << std::endl;
      intgExpFace_(scalar_index, 1) = 1.0 - quadrature_.integration_point_location(k,j);
      std::cout << "intgExpFace_(" << scalar_index << ", " << 1 << ") = " << intgExpFace_(scalar_index, 1) << std::endl;

      ++scalar_index;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 2; //left face
  oppFaceIndex = 0;
  //NOTE: this face is reversed
  int elemNodeM1 = static_cast<int>(nodes1D_-1);
  for (int k = elemNodeM1; k >= 0; --k) {
    const int nearNode = face_node_number(nodes1D_-k-1,faceOrdinal);
    int oppNode = nodeMap(k,1);
    std::cout << "nearNode: " << nearNode << std::endl;
    std::cout << "oppNode: " << oppNode << std::endl;
    
    for (int j = quadrature_.num_quad()-1; j >= 0; --j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
//      oppFace_(scalar_index) = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_; // Not sure what this is for

      intgExpFace_(scalar_index, 0) = 0.0;
      std::cout << "intgExpFace_(" << scalar_index << ", " << 0 << ") = " << intgExpFace_(scalar_index, 0) << std::endl;
      intgExpFace_(scalar_index, 1) = quadrature_.integration_point_location(k,j);
      std::cout << "intgExpFace_(" << scalar_index << ", " << 1 << ") = " << intgExpFace_(scalar_index, 1) << std::endl;

      ++scalar_index;
      ++oppFaceIndex;
    }
  }
}

void
HigherOrderTri2DSCS::shape_fcn(double *shpfc)
{
  const int polyOrder = nodes1D_ - 1;
  
  if (polyOrder == 1) {
    tri_shape_fcn_p1(numIntPoints_, intgLoc_, shpfc);
  }
  else if (polyOrder == 2) {
    tri_shape_fcn_p2(numIntPoints_, intgLoc_, shpfc);
  }
  else {
    ThrowErrorMsg("Shape functions not defined for the chosen polyOrder");
  }
}

void HigherOrderTri2DSCS::tri_shape_fcn_p1(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int threej = 3*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    shape_fcn[threej] = 1.0 - xi - eta;
    shape_fcn[1 + threej] = xi;
    shape_fcn[2 + threej] = eta;
  }
}

void HigherOrderTri2DSCS::tri_shape_fcn_p2(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int sixj = 6*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    shape_fcn[sixj] = (1-xi-eta)*(1-2.0*xi-2.0*eta);
    shape_fcn[1 + sixj] = xi*(2.0*xi-1);
    shape_fcn[2 + sixj] = eta*(2.0*eta-1);
    shape_fcn[3 + sixj] = 4.0*xi*(1-xi-eta);
    shape_fcn[4 + sixj] = 4.0*xi*eta;
    shape_fcn[5 + sixj] = 4.0*eta*(1-xi-eta);
  }
}

void HigherOrderTri2DSCS::tri_deriv_shape_fcn_p1(
  const int   npts, 
  double *deriv)
{
  for (int j = 0; j < npts; ++j ) {
    const int sixj = 6*j;
    deriv[sixj] =     -1.0;   // IP j, Node 0, dxi
    deriv[1 + sixj] = -1.0;   // IP j, Node 0, deta
    deriv[2 + sixj] =  1.0;   // IP j, Node 1, dxi
    deriv[3 + sixj] =  0.0;   // IP j, Node 1, deta
    deriv[4 + sixj] =  0.0;   // IP j, Node 2, dxi
    deriv[5 + sixj] =  1.0;   // IP j, Node 2, deta
  }
}

void HigherOrderTri2DSCS::tri_deriv_shape_fcn_p2(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *deriv)
{
  for (int j = 0; j < npts; ++j ) {
    const int twelvej = 12*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    deriv[twelvej] =       4.0*eta+4.0*xi-3.0;     // IP j, Node 0, dxi
    deriv[1 + twelvej] =   4.0*eta+4.0*xi-3.0;     // IP j, Node 0, deta
    deriv[2 + twelvej] =   4.0*xi-1.0;             // IP j, Node 1, dxi
    deriv[3 + twelvej] =   0.0;                    // IP j, Node 1, deta
    deriv[4 + twelvej] =   0.0;                    // IP j, Node 2, dxi
    deriv[5 + twelvej] =   4.0*eta-1.0;            // IP j, Node 2, deta
    deriv[6 + twelvej] =  -4.0*(eta+2.0*xi-1.0);   // IP j, Node 3, dxi
    deriv[7 + twelvej] =  -4.0*xi;                 // IP j, Node 3, deta
    deriv[8 + twelvej] =   4.0*eta;                // IP j, Node 4, dxi
    deriv[9 + twelvej] =   4.0*xi;                 // IP j, Node 4, deta
    deriv[10 + twelvej] = -4.0*eta;                // IP j, Node 5, dxi
    deriv[11 + twelvej] = -4.0*(2.0*eta+xi-1);     // IP j, Node 5, deta
  }
}

const int *
HigherOrderTri2DSCS::ipNodeMap(int ordinal) const
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_(ordinal*ipsPerFace_);
}

const int *
HigherOrderTri2DSCS::side_node_ordinals(int ordinal) const
{
  return &sideNodeOrdinals_(ordinal,0);
}

void
HigherOrderTri2DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");
  const int polyOrder = nodes1D_ - 1;
  const int ipsPerSubface = quadrature_.num_quad();
  const int numSubfaces = numIntPoints_ / quadrature_.num_quad();
  Kokkos::View<double**> realCoords;
  Kokkos::View<double**> isoParCoords;

  // Loop through all internal faces
  int offset = 0;
  for (int face = 0; face < numSubfaces; ++face) {
    std::cout << "face: " << face << std::endl;
    realCoords = Kokkos::View<double[2][2]>("realCoords");
    isoParCoords = Kokkos::View<double[2][2]>("isoParCoords");
    realCoords(0, 0) = 0.0;
    realCoords(0, 1) = 0.0;
    realCoords(1, 0) = 0.0;
    realCoords(1, 1) = 0.0;
    isoParCoords(0, 0) = intSubfaces_(face, 0, 0);
    isoParCoords(0, 1) = intSubfaces_(face, 0, 1);
    isoParCoords(1, 0) = intSubfaces_(face, 1, 0);
    isoParCoords(1, 1) = intSubfaces_(face, 1, 1);
    
    std::vector<double> shape_fcn(2 * nodesPerElement_);
    double *p_shape_fcn = &shape_fcn[0];

    // Evaluate shape functions at the endpoints of the subface (B and P)
    if (polyOrder == 1) {
      tri_shape_fcn_p1(2, isoParCoords, &p_shape_fcn[0]);
    }
    else if (polyOrder == 2) {
      tri_shape_fcn_p2(2, isoParCoords, &p_shape_fcn[0]);
    }
    else {
    ThrowErrorMsg("Shape functions not defined for the chosen polyOrder");
    }
    
    // Use isoparametric mapping to get real coordinates of B and P
    int count = 0;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < nodesPerElement_; ++j) {
        realCoords(i, 0) += (shape_fcn[count] * coords[j * nDim_ + 0]);
        std::cout << "realCoords(" << i << ", 0) += " << "shape_fcn[" << count << "] * coords[" << j << " * " << nDim_ << " + 0]" << std::endl;
        realCoords(i, 1) += (shape_fcn[count] * coords[j * nDim_ + 1]);
        std::cout << "realCoords(" << i << ", 1) += " << "shape_fcn[" << count << "] * coords[" << j << " * " << nDim_ << " + 1]" << std::endl;
        count++;
      }
    }

    const double Bx = realCoords(0, 0);
    const double By = realCoords(0, 1);
    const double Px = realCoords(1, 0);
    const double Py = realCoords(1, 1);
    std::cout << "Bx = " << Bx << ", By = " << By << ", Px = " << Px << ", Py = " << Py << std::endl;

    const double dx = Px - Bx;
    const double dy = Py - By;

    // Loop through all IPs of the current subface
    for (int ip = 0; ip < ipsPerSubface; ++ip) {
      const double weight = ipWeights_(offset + ip);
      std::cout << "ipWeights_(" << offset+ip << ") = " << weight << std::endl;
      
      areav[(offset + ip) * nDim_ + 0] =  dy * weight;
      std::cout << "areav[" << (offset + ip) * nDim_ + 0 << "] = " << dy * weight << std::endl;
      areav[(offset + ip) * nDim_ + 1] = -dx * weight;
      std::cout << "areav[" << (offset + ip) * nDim_ + 1 << "] = " << -dx * weight << std::endl;
    }
    offset += ipsPerSubface;
  }
}

void HigherOrderTri2DSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "Grad_op is executed one element at a time for HO");

  int grad_offset = 0;
  int grad_inc = nDim_ * nodesPerElement_;
  const int polyOrder = nodes1D_ - 1;
  
  if (polyOrder == 1) {
    tri_deriv_shape_fcn_p1(numIntPoints_, deriv);
  }
  else if (polyOrder == 2) {
    tri_deriv_shape_fcn_p2(numIntPoints_, intgLoc_, deriv);
  }
  else {
    ThrowErrorMsg("Shape function derivatives not defined for the chosen polyOrder");
  }

  for (int ip = 0; ip < numIntPoints_; ++ip) {

    gradient_2d(
      nodesPerElement_,
      &coords[0],
      &deriv[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
    );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}

void
HigherOrderTri2DSCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "face_grad_op is executed one element at a time for HO");

  int grad_offset = 0;
  int grad_inc = nDim_ * nodesPerElement_;

  const int face_offset =  nDim_ * ipsPerFace_ * nodesPerElement_ * face_ordinal;
  const double* const faceShapeDerivs = &expFaceShapeDerivs_.data()[face_offset];

  for (int ip = 0; ip < ipsPerFace_; ++ip) {
    gradient_2d(
      nodesPerElement_,
      coords,
      &faceShapeDerivs[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
   );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}


const int *
HigherOrderTri2DSCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_.data();
}

int
HigherOrderTri2DSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_(ordinal*ipsPerFace_+node);
}

int
HigherOrderTri2DSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_(ordinal*ipsPerFace_+node);
}

template <Jacobian::Direction direction> void
HigherOrderTri2DSCS::area_vector(
  const double *POINTER_RESTRICT elemNodalCoords,
  double *POINTER_RESTRICT shapeDeriv,
  double *POINTER_RESTRICT normalVec ) const
{
  constexpr int s1Component = (direction == Jacobian::S_DIRECTION) ?
      Jacobian::T_DIRECTION : Jacobian::S_DIRECTION;

  double dxdr = 0.0;  double dydr = 0.0;
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];

    dxdr += shapeDeriv[vector_offset+s1Component] * xCoord;
    dydr += shapeDeriv[vector_offset+s1Component] * yCoord;
  }

  normalVec[0] =  dydr;
  normalVec[1] = -dxdr;
}

void HigherOrderTri2DSCS::gij(
  const double *coords,
  double *gupperij,
  double *glowerij,
  double *deriv)
{
  SIERRA_FORTRAN(twod_gij)
    ( &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gupperij, glowerij);
}

} // namespace nalu
} // namespace Sierra