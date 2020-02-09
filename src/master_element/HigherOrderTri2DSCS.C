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

namespace sierra{
namespace nalu{

KOKKOS_FUNCTION
HigherOrderTri2DSCS::HigherOrderTri2DSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature)),
  numQuad_(quadrature.num_quad())
#ifndef KOKKOS_ENABLE_CUDA
  , nodeMap(make_node_map_tri(basis.order())),
  faceNodeMap(make_face_node_map_tri(basis.order())),
  sideNodeOrdinals_(make_side_node_ordinal_map_tri(basis.order()))
#endif
  , nodes1D_(basis.order()+1),
  polyOrder_(nodes1D_-1)
{
  MasterElement::nDim_ = 2;
  nodesPerElement_ = 0.5*(nodes1D_*(nodes1D_+1));

#ifndef KOKKOS_ENABLE_CUDA
  // set up integration rule and relevant maps for scs
  set_interior_info();

  // set up integration rule and relevant maps for faces
  set_boundary_info();
#endif
}

std::vector<double> 
HigherOrderTri2DSCS::getCentroid(std::vector<ordinal_type> nodeOrdinals, std::unique_ptr<ElementDescription>& eleDesc) {
  const double length = (double)nodeOrdinals.size();
  const double factor = 1.0/length;
  std::vector<double> centroid(2, 0.0);
  for (auto nodeOrdinal : nodeOrdinals) {
    for (int i = 0; i < 2; ++i) {        
      const double coord = eleDesc->nodeLocs[nodeOrdinal][i];
      centroid[i] += factor * coord;
    }
  }
  
  return centroid;
}

// Set integration locations in the isoparametric coordinate frame and
// the left/right node ordinal to each integration point
void
HigherOrderTri2DSCS::set_interior_info()
{
  auto desc = ElementDescription::create(2, polyOrder_, stk::topology::TRI_3_2D);
  
  int IPCount = 0;
  int subfaceCount = 0;
  std::vector<double> subTriCentroid(2);
  std::vector<double> bottomEdgeCentroid(2);
  std::vector<double> rightEdgeCentroid(2);
  std::vector<double> leftEdgeCentroid(2);
  ordinal_type leftNode;
  ordinal_type rightNode;

  // lambda to compute the integration location and weight
  auto writeIPInfo = [&](int& subFC, int& count, std::vector<double> subTriCentroid, std::vector<double> subEdgeCentroid, ordinal_type left, ordinal_type right, int orientation) {
    
    // Save BP vector (subface) for later usage in areav computation
    // Note that the coords are in the isoparametric coord frame
    intSubfaces_(subFC, 0, 0) = subTriCentroid[0];
    intSubfaces_(subFC, 0, 1) = subTriCentroid[1];
    intSubfaces_(subFC, 1, 0) = subEdgeCentroid[0];
    intSubfaces_(subFC, 1, 1) = subEdgeCentroid[1];
    subFC++;
    
    const double scsLength_x = subEdgeCentroid[0] - subTriCentroid[0];
    const double scsLength_y = subEdgeCentroid[1] - subTriCentroid[1];
    const double scsLength = std::sqrt(scsLength_x*scsLength_x + scsLength_y*scsLength_y);
    double xl = 0;
    double xr = scsLength;
    
    for (int j = 0; j < numQuad_; ++j) {
      const double abscissa = quadrature_.abscissa(j);
      ipWeights_(count) = orientation * quadrature_.weights(j);
      lrscv_(2*count) = left;
      lrscv_(2*count+1) = right;
      
      const double intgLoc1D = 0.5*( abscissa*(xr-xl)+(xl+xr));
      for (int i = 0; i < 2; ++i) {
        intgLoc_(count, i) = subTriCentroid[i] + (intgLoc1D/scsLength)*( subEdgeCentroid[i]-subTriCentroid[i] );
      }
      count++;
    }
  };

  // Hardcode left/right node mapping and integration locations
  if (polyOrder_ == 1) {
    numIntPoints_ = 3;
    int orientation;

    // define L/R mappings
    lrscv_ = Kokkos::View<int*>("lsrcv_", 2*numIntPoints_);

    // standard integration location
    intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
    ipWeights_ = Kokkos::View<double*>("ipWeights_", numIntPoints_);
    intSubfaces_ = Kokkos::View<double*[2][2]>("intSubfaces_", numIntPoints_/numQuad_);
    
    subTriCentroid = getCentroid({0, 1, 2}, desc);
    bottomEdgeCentroid = getCentroid({0, 1}, desc);
    rightEdgeCentroid = getCentroid({1, 2}, desc);
    leftEdgeCentroid = getCentroid({2, 0}, desc);
    
    // bottom edge face endloc
    orientation = -1;
    leftNode = 0;
    rightNode = 1;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, bottomEdgeCentroid, leftNode, rightNode, orientation);
    
    // right edge face endloc
    orientation = -1;
    leftNode = 1;
    rightNode = 2;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, rightEdgeCentroid, leftNode, rightNode, orientation);
    
    // left edge face endloc
    orientation = 1;
    leftNode = 0;
    rightNode = 2;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, leftEdgeCentroid, leftNode, rightNode, orientation);
  }
  else if (polyOrder_ == 2) {
    numIntPoints_ = 24;
    int orientation;

    // define L/R mappings
    lrscv_ = Kokkos::View<int*>("lsrcv_", 2*numIntPoints_);

    // standard integration location
    intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
    ipWeights_ = Kokkos::View<double*>("ipWeights_", numIntPoints_);
    intSubfaces_ = Kokkos::View<double*[2][2]>("intSubfaces_", numIntPoints_/numQuad_);
    
    //----------------------------------------------------------------
    // bottom left subtriangle
    subTriCentroid = getCentroid({0, 3, 5}, desc);
    bottomEdgeCentroid = getCentroid({0, 3}, desc);
    rightEdgeCentroid = getCentroid({3, 5}, desc);
    leftEdgeCentroid = getCentroid({5, 0}, desc);
    
    // bottom edge face endloc
    orientation = -1;
    leftNode = 0;
    rightNode = 3;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, bottomEdgeCentroid, leftNode, rightNode, orientation);
    
    // right edge face endloc
    orientation = -1;
    leftNode = 3;
    rightNode = 5;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, rightEdgeCentroid, leftNode, rightNode, orientation);
    
    // left edge face endloc
    orientation = 1;
    leftNode = 0;
    rightNode = 5;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, leftEdgeCentroid, leftNode, rightNode, orientation);
    
    //----------------------------------------------------------------
    // bottom right subtriangle
    subTriCentroid = getCentroid({3, 1, 4}, desc);
    bottomEdgeCentroid = getCentroid({3, 1}, desc);
    rightEdgeCentroid = getCentroid({1, 4}, desc);
    leftEdgeCentroid = getCentroid({4, 3}, desc);
    
    // bottom edge face endloc
    orientation = 1;
    leftNode = 1;
    rightNode = 3;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, bottomEdgeCentroid, leftNode, rightNode, orientation);
    
    // right edge face endloc
    orientation = -1;
    leftNode = 1;
    rightNode = 4;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, rightEdgeCentroid, leftNode, rightNode, orientation);
    
    // left edge face endloc
    orientation = 1;
    leftNode = 3;
    rightNode = 4;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, leftEdgeCentroid, leftNode, rightNode, orientation);
    
    //----------------------------------------------------------------
    // top subtriangle
    subTriCentroid = getCentroid({5, 4, 2}, desc);
    bottomEdgeCentroid = getCentroid({5, 4}, desc);
    rightEdgeCentroid = getCentroid({4, 2}, desc);
    leftEdgeCentroid = getCentroid({2, 5}, desc);
    
    // bottom edge face endloc
    orientation = 1;
    leftNode = 4;
    rightNode = 5;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, bottomEdgeCentroid, leftNode, rightNode, orientation);
    
    // right edge face endloc
    orientation = 1;
    leftNode = 2;
    rightNode = 4;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, rightEdgeCentroid, leftNode, rightNode, orientation);
    
    // left edge face endloc
    orientation = -1;
    leftNode = 2;
    rightNode = 5;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, leftEdgeCentroid, leftNode, rightNode, orientation);
    
    //----------------------------------------------------------------
    // middle subtriangle
    subTriCentroid = getCentroid({3, 4, 5}, desc);
    bottomEdgeCentroid = getCentroid({3, 4}, desc);
    rightEdgeCentroid = getCentroid({4, 5}, desc);
    leftEdgeCentroid = getCentroid({5, 3}, desc);
    
    // bottom edge face endloc
    orientation = -1;
    leftNode = 3;
    rightNode = 4;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, bottomEdgeCentroid, leftNode, rightNode, orientation);
    
    // right edge face endloc
    orientation = -1;
    leftNode = 4;
    rightNode = 5;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, rightEdgeCentroid, leftNode, rightNode, orientation);
    
    // left edge face endloc
    orientation = 1;
    leftNode = 3;
    rightNode = 5;
    writeIPInfo(subfaceCount, IPCount, subTriCentroid, leftEdgeCentroid, leftNode, rightNode, orientation);
  }
  else {
    ThrowErrorMsg("Only P1 and P2 is defined for TRI_2_2D elements.");
  }
}

// Not yet adapted to new (analog to tet) scs creation, still follows the quad style
// and also the usage of this function is unclear
void
HigherOrderTri2DSCS::set_boundary_info()
{
//  const int numFaces = 3;
//  const int nodesPerFace = nodes1D_;
//  ipsPerFace_ = nodesPerFace*numQuad_;
//
//  const int numFaceIps = numFaces*ipsPerFace_;
//
//  oppFace_ =Kokkos::View<int*>("oppFace_", numFaceIps);
//  ipNodeMap_ = Kokkos::View<int*>("ipNodeMap_", numFaceIps);
//  oppNode_ = Kokkos::View<int*>("oppNode", numFaceIps);
//  intgExpFace_=Kokkos::View<double**>("intgExpFace_", numFaceIps,nDim_);
//
//
//  auto face_node_number = [&] (int number,int faceOrdinal)
//  {
//    return faceNodeMap(faceOrdinal,number);
//  };
//
//  int scalar_index = 0;
//  int faceOrdinal = 0; //bottom face
//  int oppFaceIndex = 0;
//  for (int k = 0; k < nodes1D_; ++k) {
//    const int nearNode = face_node_number(k,faceOrdinal);
//    int oppNode = nodeMap(k, 1);
////    std::cout << "nearNode: " << nearNode << std::endl;
////    std::cout << "oppNode: " << oppNode << std::endl;
//
//    for (int j = 0; j < numQuad_; ++j) {
//      ipNodeMap_(scalar_index) = nearNode;
//      oppNode_(scalar_index) = oppNode;
////      oppFace_(scalar_index) = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_; // Not sure what this is for
//      
//      intgExpFace_(scalar_index, 0) = quadrature_.integration_point_location(k,j);
////      std::cout << "intgExpFace_(" << scalar_index << ", " << 0 << ") = " << intgExpFace_(scalar_index, 0) << std::endl;
//      intgExpFace_(scalar_index, 1) = 0.0;
////      std::cout << "intgExpFace_(" << scalar_index << ", " << 1 << ") = " << intgExpFace_(scalar_index, 1) << std::endl;
//
//      ++scalar_index;
//      ++oppFaceIndex;
//    }
//  }
//
//  faceOrdinal = 1; //right face
//  oppFaceIndex = 0;
//  for (int k = 0; k < nodes1D_; ++k) {
//    const int nearNode = face_node_number(k,faceOrdinal);
//    int oppNode = nodeMap(k, nodes1D_-2);
////    std::cout << "nearNode: " << nearNode << std::endl;
////    std::cout << "oppNode: " << oppNode << std::endl;
//
//    for (int j = 0; j < quadrature_.num_quad(); ++j) {
//      ipNodeMap_(scalar_index) = nearNode;
//      oppNode_(scalar_index) = oppNode;
////      oppFace_(scalar_index) = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_; // Not sure what this is for
//
//      intgExpFace_(scalar_index, 0) = quadrature_.integration_point_location(k,j);
////      std::cout << "intgExpFace_(" << scalar_index << ", " << 0 << ") = " << intgExpFace_(scalar_index, 0) << std::endl;
//      intgExpFace_(scalar_index, 1) = 1.0 - quadrature_.integration_point_location(k,j);
////      std::cout << "intgExpFace_(" << scalar_index << ", " << 1 << ") = " << intgExpFace_(scalar_index, 1) << std::endl;
//
//      ++scalar_index;
//      ++oppFaceIndex;
//    }
//  }
//
//  faceOrdinal = 2; //left face
//  oppFaceIndex = 0;
//  //NOTE: this face is reversed
//  int elemNodeM1 = static_cast<int>(nodes1D_-1);
//  for (int k = elemNodeM1; k >= 0; --k) {
//    const int nearNode = face_node_number(nodes1D_-k-1,faceOrdinal);
//    int oppNode = nodeMap(k,1);
////    std::cout << "nearNode: " << nearNode << std::endl;
////    std::cout << "oppNode: " << oppNode << std::endl;
//    
//    for (int j = quadrature_.num_quad()-1; j >= 0; --j) {
//      ipNodeMap_(scalar_index) = nearNode;
//      oppNode_(scalar_index) = oppNode;
////      oppFace_(scalar_index) = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_; // Not sure what this is for
//
//      intgExpFace_(scalar_index, 0) = 0.0;
////      std::cout << "intgExpFace_(" << scalar_index << ", " << 0 << ") = " << intgExpFace_(scalar_index, 0) << std::endl;
//      intgExpFace_(scalar_index, 1) = quadrature_.integration_point_location(k,j);
////      std::cout << "intgExpFace_(" << scalar_index << ", " << 1 << ") = " << intgExpFace_(scalar_index, 1) << std::endl;
//
//      ++scalar_index;
//      ++oppFaceIndex;
//    }
//  }
}

void
HigherOrderTri2DSCS::shape_fcn(double *shpfc)
{
  if (polyOrder_ == 1) {
    tri_shape_fcn_p1(numIntPoints_, intgLoc_, shpfc);
  }
  else if (polyOrder_ == 2) {
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
    shape_fcn[sixj] = (1.0-xi-eta)*(1.0-2.0*xi-2.0*eta);
    shape_fcn[1 + sixj] = xi*(2.0*xi-1.0);
    shape_fcn[2 + sixj] = eta*(2.0*eta-1.0);
    shape_fcn[3 + sixj] = 4.0*xi*(1.0-xi-eta);
    shape_fcn[4 + sixj] = 4.0*xi*eta;
    shape_fcn[5 + sixj] = 4.0*eta*(1.0-xi-eta);
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
    deriv[11 + twelvej] = -4.0*(2.0*eta+xi-1.0);   // IP j, Node 5, deta
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
  const int ipsPerSubface = numQuad_;
  const int numSubfaces = numIntPoints_/numQuad_;
  Kokkos::View<double**> realCoords;
  Kokkos::View<double**> isoParCoords;

  // Loop through all internal faces
  int offset = 0;
  for (int face = 0; face < numSubfaces; ++face) {
//    std::cout << "face: " << face << std::endl;
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
        realCoords(i, 1) += (shape_fcn[count] * coords[j * nDim_ + 1]);
        count++;
      }
//      std::cout << "realCoord_x = " << realCoords(i, 0) << ", realCoord_y = " << realCoords(i, 1) << std::endl;
    }

    const double Bx = realCoords(0, 0);
    const double By = realCoords(0, 1);
    const double Px = realCoords(1, 0);
    const double Py = realCoords(1, 1);
//    std::cout << "Bx = " << Bx << ", By = " << By << ", Px = " << Px << ", Py = " << Py << std::endl;

    const double dx = Px - Bx;
    const double dy = Py - By;

    // Loop through all IPs of the current subface
    for (int ip = 0; ip < ipsPerSubface; ++ip) {
      const double weight = ipWeights_(offset + ip);
      
      // 90° rotation of the vector {dx, dy} to get area normal vector
      areav[(offset + ip) * nDim_ + 0] =  dy * weight;
      areav[(offset + ip) * nDim_ + 1] = -dx * weight;
        
//      std::cout << std::endl;
//      std::cout << "new IP" << std::endl;
//      std::cout << "areav = {" << dy*weight << ", " << -dx*weight << "}" << std::endl;
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


const int *
HigherOrderTri2DSCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_.data();
}

} // namespace nalu
} // namespace Sierra