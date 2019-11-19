/* 
 * File:   HigherOrderTetSCS.C
 * Author: Raphael Lindegger
 * 
 * Created on November 2, 2019, 12:49 PM
 */

#include <master_element/HigherOrderTetSCS.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/HigherOrderMasterElementFunctions.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/TensorOps.h>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/NodeMapMaker.h>
#include <element_promotion/TensorProductQuadratureRule.h>

#include <NaluEnv.h>
#include <master_element/MasterElement.h>
#include <FORTRAN_Proto.h>

#include <BuildTemplates.h>

#include <stk_util/util/ReportHandler.hpp>

#include <array>
#include <limits>
#include <cmath>
#include <memory>
#include <stdexcept>


namespace sierra{
namespace nalu{

KOKKOS_FUNCTION
HigherOrderTetSCS::HigherOrderTetSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  nodes1D_(basis.order() + 1),
  numQuad_(quadrature.num_quad()*quadrature.num_quad()),
  numSubsurfacesPerSubelement_(6),
  numSubsurfacesPerSubface_(3),
  polyOrder_(nodes1D_-1),
#ifndef KOKKOS_ENABLE_CUDA
  nodeMap(make_node_map_tet(basis.order(), true)),
  faceNodeMap(make_face_node_map_tet(basis.order())),
  sideNodeOrdinals_(make_side_node_ordinal_map_tet(basis.order())),
#endif
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature))
#ifndef KOKKOS_ENABLE_CUDA
  , expRefGradWeights_("reference_gradient_weights", 1, basis.num_nodes())
#endif
{
  MasterElement::nDim_ = 3;
  nodesPerElement_ = (polyOrder_+3)*(polyOrder_+2)*(polyOrder_+1)/6; // Tetrahedral number

#ifndef KOKKOS_ENABLE_CUDA
  // generate hex shape functions used for the isoparametric mapping intgLoc on subsurfaces (scs)
  intgLocSurfIso_ = Kokkos::View<double**>("integration_point_location_subsurf", numQuad_, 3);
  if (polyOrder_ == 1) {
    // define IP location in isoparametric subsurface
    // IP1, there is just one for P1
    intgLocSurfIso_(0, 0) = quadrature_.abscissa(0);
    intgLocSurfIso_(0, 1) = quadrature_.abscissa(0);
    intgLocSurfIso_(0, 2) = 0; 
  }
  else if (polyOrder_ == 2) {
    // define IP locations in isoparametric subsurface
    // IP1
    intgLocSurfIso_(0, 0) = quadrature_.abscissa(0);
    intgLocSurfIso_(0, 1) = quadrature_.abscissa(0);
    intgLocSurfIso_(0, 2) = 0;
    // IP2
    intgLocSurfIso_(1, 0) = quadrature_.abscissa(1);
    intgLocSurfIso_(1, 1) = quadrature_.abscissa(0);
    intgLocSurfIso_(1, 2) = 0;
    // IP3
    intgLocSurfIso_(2, 0) = quadrature_.abscissa(0);
    intgLocSurfIso_(2, 1) = quadrature_.abscissa(1);
    intgLocSurfIso_(2, 2) = 0;
    // IP4
    intgLocSurfIso_(3, 0) = quadrature_.abscissa(1);
    intgLocSurfIso_(3, 1) = quadrature_.abscissa(1);
    intgLocSurfIso_(3, 2) = 0;
  }
  else {
    ThrowErrorMsg("Only P1 and P2 is defined for TET_4 elements.");
  }
  
  shape_fcnHex_.resize(numQuad_ * 8);
  double *p_shape_fcnHex = &shape_fcnHex_[0];
  hex_shape_fcn_p1(numQuad_, intgLocSurfIso_, &p_shape_fcnHex[0]);
  
  // set up integration rule and relevant maps on scs
  set_interior_info();

  // set up integration rule and relevant maps on faces
  set_boundary_info();

  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
  expFaceShapeDerivs_ = basis_.eval_deriv_weights(intgExpFace_);
#endif
}

std::vector<double> getCentroid(std::vector<ordinal_type>& nodeOrdinals, std::unique_ptr<ElementDescription>& eleDesc) {
  const double length = (double)nodeOrdinals.size();
  const double factor = 1.0/length;
  std::vector<double>centroid(3, 0.0);
  for (auto nodeOrdinal : nodeOrdinals) {
    for (int i = 0; i < 3; ++i) {        
      const double coord = eleDesc->nodeLocs[nodeOrdinal][i];
      centroid[i] += factor * coord;
    }
  }
  
  return centroid;
}

void
HigherOrderTetSCS::hex_shape_fcn_p1(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int eightj = 8*j;
    const double oneEighth = 1.0/8.0;
    const double xi   = par_coord(j, 0);
    const double eta  = par_coord(j, 1);
    const double zeta = par_coord(j, 2);
    shape_fcn[0 + eightj] = oneEighth*(1.0-xi)*(1.0-eta)*(1.0-zeta);
    shape_fcn[1 + eightj] = oneEighth*(1.0+xi)*(1.0-eta)*(1.0-zeta);
    shape_fcn[2 + eightj] = oneEighth*(1.0+xi)*(1.0+eta)*(1.0-zeta);
    shape_fcn[3 + eightj] = oneEighth*(1.0-xi)*(1.0+eta)*(1.0-zeta);
    shape_fcn[4 + eightj] = oneEighth*(1.0-xi)*(1.0-eta)*(1.0+zeta);
    shape_fcn[5 + eightj] = oneEighth*(1.0+xi)*(1.0-eta)*(1.0+zeta);
    shape_fcn[6 + eightj] = oneEighth*(1.0+xi)*(1.0+eta)*(1.0+zeta);
    shape_fcn[7 + eightj] = oneEighth*(1.0-xi)*(1.0+eta)*(1.0+zeta);
  }
}

void
HigherOrderTetSCS::set_interior_info()
{
  auto desc = ElementDescription::create(3, polyOrder_, stk::topology::TET_4);
  
  numSubelements_ = desc->subElementConnectivity.size();
  numIntPoints_ = numSubelements_ * numSubsurfacesPerSubelement_ * numQuad_;
  lrscv_ = Kokkos::View<int**>("left_right_state_mapping", numIntPoints_, 2);
  intgLoc_ = Kokkos::View<double**>("integration_point_location", numIntPoints_, 3);
  ipWeights_ = Kokkos::View<double*>("ip_weight", numIntPoints_);
  subsurfaceNodeLoc_.resize(numSubelements_ * numSubsurfacesPerSubelement_ * 4, std::vector<double>(3));

  ordinal_type left;
  ordinal_type right;
  
  int countIP = 0;
  
  std::vector<std::vector<int>> subsurfCreationIndices {
    {0, 1, 2, 3}, // element centroid
    {0, 1, 2}, // subface centroid 1
    {0, 1}, // subedge centroid
    {0, 1, 3}, //subface centroid 2
    {0, 1, 2, 3}, // element centroid again because of simplicity
    {0, 1, 2},
    {1, 2},
    {1, 2, 3},   
    {0, 1, 2, 3}, // element centroid again because of simplicity
    {0, 1, 2},
    {0, 2},
    {0, 2, 3}, 
    {0, 1, 2, 3}, // element centroid again because of simplicity
    {0, 1, 3},
    {1, 3},
    {1, 2, 3},
    {0, 1, 2, 3}, // element centroid again because of simplicity
    {0, 1, 3},
    {0, 3},
    {0, 2, 3},
    {0, 1, 2, 3}, // element centroid again because of simplicity
    {0, 2, 3},
    {2, 3},
    {1, 2, 3}
  };

  // initialize intgLoc_
  for (int i = 0; i < numIntPoints_; ++i) {
    for (int j = 0; j < nDim_; ++j) {
      intgLoc_(i, j) = 0.0;
    }
  }
   
  // loop through each subelement and compute the integration points at each subsurface in the subelement
  int countNode = 0;
  for (int subElement = 0; subElement < numSubelements_; ++subElement) {
    
    int countSubsurf = 0;
    for (int subSurf = 0; subSurf < 6; ++subSurf) {
      
      for (int node = 0; node < 4; ++node) {
        const int numOrd = subsurfCreationIndices[countSubsurf].size();
        std::vector<ordinal_type> centroidDefiningOrdinals(numOrd);
        
        for (int i = 0; i < numOrd; ++i) {
          const int ordIndex = subsurfCreationIndices[countSubsurf][i];
          centroidDefiningOrdinals[i] = desc->subElementConnectivity[subElement][ordIndex];
        }
        
        // compute subsurface node location and save it for later usage in areav computation
        std::vector<double> nodeLoc = getCentroid(centroidDefiningOrdinals, desc);
        
        int subsurfaceNodeLocIndex = 24*subElement + 4*subSurf + node;
        subsurfaceNodeLoc_[subsurfaceNodeLocIndex] = nodeLoc;
        
        // if current ordinals describe a subedge (only 2 ordinals), use them for the left/right node mapping
        if (node == 2) {
          left = centroidDefiningOrdinals[0];
          right = centroidDefiningOrdinals[1];
        }
        
        countSubsurf++;
      }

      // isoparametric mapping of the intgLoc of a isoparametric rectangle to the isoparametric tet
      int countHexSF = 0;
      for (int quadPoint = 0; quadPoint < numQuad_; ++quadPoint) { // for each ip at subsurf
        std::cout << "new quadpoint" << std::endl;

        // IP weight
        int orientation = 1;
        ipWeights_(countIP) = orientation * quadrature_.weights(quadPoint) * quadrature_.weights(quadPoint);

        // left/right node mapping
        lrscv_(countIP, 0) = left;
        lrscv_(countIP, 1) = right;
        std::cout << "left node: " << lrscv_(countIP, 0) << ", right node: " << lrscv_(countIP, 1) << std::endl;

        for (int k = 0; k < 2; ++k) { // repeat 2 times because hex shape functions have 8 nodes but the subsurf has 4 nodes
          
          for (int i = 0; i < 4; ++i) { // for each node of the subsurf
            int subsurfaceNodeLocIndex = 24*subElement + 4*subSurf + i;
            
            for (int j = 0; j < 3; ++j) { // for each dimension
              intgLoc_(countIP, j) += (shape_fcnHex_[countHexSF] * subsurfaceNodeLoc_[subsurfaceNodeLocIndex][j]);
            }
            
            countHexSF++;
          }
        }
        
        std::cout << "isoCalc intgLoc: " << intgLoc_(countIP, 0) << ", " << intgLoc_(countIP, 1) << ", " << intgLoc_(countIP, 2) << std::endl;
        countIP++;
        
      } // ip
    } // subSurf
  } // subElement
}

// copied from hex and not yet adapted to tet
int HigherOrderTetSCS::opposing_face_map(int k, int l, int i, int j, int face_index)
{
  const int surfacesPerDirection = nodes1D_ - 1;
  const int faceToSurface[6] = {
      surfacesPerDirection,     // nearest scs face to t=-1.0
      3*surfacesPerDirection-1, // nearest scs face to s=+1.0, the last face
      2*surfacesPerDirection-1, // nearest scs face to t=+1.0
      2*surfacesPerDirection,   // nearest scs face to s=-1.0
      0,                        // nearest scs face to u=-1.0, the first face
      surfacesPerDirection-1    // nearest scs face to u=+1.0, the first face
  };

  const int face_offset = faceToSurface[face_index] * ipsPerFace_;
  const int node_index = k + nodes1D_ * l;
  const int node_offset = node_index * (numQuad_ * numQuad_);
  const int ip_index = face_offset + node_offset + i + numQuad_ * j;

  return ip_index;
}

void
HigherOrderTetSCS::set_boundary_info()
{
  const int numFaces = 4;
  const int nodesPerFace = 0.5*(nodes1D_*(nodes1D_+1)); // triangular number
  ipsPerFace_ = nodesPerFace * numQuad_ * numQuad_;
  
  const int numFaceIps = numFaces*ipsPerFace_;
  ipNodeMap_ = Kokkos::View<int*>("owning_node_for_ip", numFaceIps);
  intgExpFace_ = Kokkos::View<double**>("exposed_face_integration_loc", numFaceIps, nDim_);
  
  auto desc = ElementDescription::create(3, polyOrder_, stk::topology::TET_4);
  int numSubfacePerFace;
  std::vector<std::vector<int>> subfaceCreationIndices;
  
  if (polyOrder_ == 1) {
    numSubfacePerFace = 1;
    subfaceCreationIndices {
      {0, 1, 2}
    };
  }
  else if (polyOrder_ == 2) {
    numSubfacePerFace = 4;
    subfaceCreationIndices {
      {0, 1, 5},
      {1, 2, 3},
      {5, 1, 3},
      {5, 3, 4}
    };
  }
  else {
    ThrowErrorMsg("Only P1 and P2 is defined for TET_4 elements.");
  }
  
  subsurfaceNodeLoc_.resize(numFaces * numSubfacePerFace * numSubsurfacesPerSubface_ * 4, std::vector<double>(3));
  
  
  // iterate through each face of the element
  for (int face = 0; face < numFaces; ++face) {
  
    // iterate through each subface of the face
    for (int subFace = 0; subFace < numSubfacePerFace; ++subFace) {
      
      std::vector<ordinal_type> subfaceOrdinals(3);
      for (int i = 0; i < 3; ++i) {
        const int subfaceNodeIndex = subfaceCreationIndices[subFace][i];
        subfaceOrdinals[i] = desc->faceNodeMap[face][subfaceNodeIndex];
      }
      
      std::vector<double> subfaceCentroid = getCentroid(subfaceOrdinals, desc);
      
      std::vector<std::vector<ordinal_type>> subedgeOrdinals = {
        {subfaceOrdinals[0], subfaceOrdinals[1]},
        {subfaceOrdinals[1], subfaceOrdinals[2]},
        {subfaceOrdinals[2], subfaceOrdinals[0]}
      };
      
      std::vector<double> subedge1Centroid = getCentroid(subedgeOrdinals[0], desc);
      std::vector<double> subedge2Centroid = getCentroid(subedgeOrdinals[1], desc);
      std::vector<double> subedge3Centroid = getCentroid(subedgeOrdinals[2], desc);
        
      for (int subSurf = 0; subSurf < 3; ++subSurf) {
        subsurfaceNodeLocBC_[subsurfaceNodeLocIndex] = nodeLoc;
      }
      
      
      // compute subface centroid
//      std::vector<double> nodeLoc = getCentroid(centroidDefiningOrdinals, desc);
      // compute subedge 1 centroid
      // compute subedge 2 centroid
      // isoparametric mapping
    }
  }
  
  
  
  
  
  
  
}

void
HigherOrderTetSCS::shape_fcn(double* shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_.data()[j];
  }
}

const int* HigherOrderTetSCS::adjacentNodes()
{
  return &lrscv_(0,0);
}

const int* HigherOrderTetSCS::ipNodeMap(int ordinal) const
{
  return &ipNodeMap_[ordinal*ipsPerFace_];
}

const int *
HigherOrderTetSCS::side_node_ordinals (int ordinal) const
{
  return &sideNodeOrdinals_(ordinal,0);
}

int
HigherOrderTetSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*ipsPerFace_+node];
}

int
HigherOrderTetSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*ipsPerFace_+node];
}

void
HigherOrderTetSCS::determinant(
  const int  /* nelem */,
  const double *coords,
  double *areav,
  double *error)
{
   constexpr int dim = 3;
   int ipsPerDirection = numIntPoints_ / dim;

   int index = 0;

   //returns the normal vector x_s x x_t for constant u surfaces
   for (int ip = 0; ip < ipsPerDirection; ++ip) {
     area_vector<Jacobian::U_DIRECTION>(coords, &shapeDerivs_(index, 0, 0), &areav[index * dim]);
     ++index;
   }

   //returns the normal vector x_u x x_s for constant t surfaces
   for (int ip = 0; ip < ipsPerDirection; ++ip) {
     area_vector<Jacobian::T_DIRECTION>(coords, &shapeDerivs_(index, 0, 0), &areav[index * dim]);
     ++index;
   }

   //returns the normal vector x_t x x_u for constant s curves
   for (int ip = 0; ip < ipsPerDirection; ++ip) {
     area_vector<Jacobian::S_DIRECTION>(coords, &shapeDerivs_(index, 0, 0), &areav[index * dim]);
     ++index;
   }

   // Multiply with the integration point weighting
   for (int ip = 0; ip < numIntPoints_; ++ip) {
     double weight = ipWeights_[ip];
     areav[ip * dim + 0] *= weight;
     areav[ip * dim + 1] *= weight;
     areav[ip * dim + 2] *= weight;
   }

   *error = 0; // no error checking available
}

template <Jacobian::Direction direction> void
HigherOrderTetSCS::area_vector(
  const double *POINTER_RESTRICT elemNodalCoords,
  double *POINTER_RESTRICT shapeDeriv,
  double *POINTER_RESTRICT areaVector) const
{
  constexpr int s1Component = (direction == Jacobian::T_DIRECTION) ?
      Jacobian::S_DIRECTION : Jacobian::T_DIRECTION;

  constexpr int s2Component = (direction == Jacobian::U_DIRECTION) ?
      Jacobian::S_DIRECTION : Jacobian::U_DIRECTION;

  // return the normal area vector given shape derivatives dnds OR dndt
  double dx_ds1 = 0.0; double dy_ds1 = 0.0; double dz_ds1 = 0.0;
  double dx_ds2 = 0.0; double dy_ds2 = 0.0; double dz_ds2 = 0.0;

  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];
    const double zCoord = elemNodalCoords[vector_offset+2];

    const double dn_ds1 = shapeDeriv[vector_offset+s1Component];
    const double dn_ds2 = shapeDeriv[vector_offset+s2Component];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
  }

  //cross product
  areaVector[0] = dy_ds1*dz_ds2 - dz_ds1*dy_ds2;
  areaVector[1] = dz_ds1*dx_ds2 - dx_ds1*dz_ds2;
  areaVector[2] = dx_ds1*dy_ds2 - dy_ds1*dx_ds2;
}

void HigherOrderTetSCS::grad_op(
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

  for (int ip = 0; ip < numIntPoints_; ++ip) {
    for (int j = 0; j < grad_inc; ++j) {
      deriv[grad_offset + j] = shapeDerivs_.data()[grad_offset +j];
    }

    gradient_3d(
      nodesPerElement_,
      coords,
      &shapeDerivs_.data()[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
    );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}

void HigherOrderTetSCS::face_grad_op(
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
    gradient_3d(
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

void HigherOrderTetSCS::gij(
  const double *coords,
  double *gupperij,
  double *glowerij,
  double *deriv)
{
  SIERRA_FORTRAN(threed_gij)
    ( &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gupperij, glowerij);
}

double HigherOrderTetSCS::isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord)
{
  std::array<double, 3> initialGuess = {{ 0.0, 0.0, 0.0 }};
  int maxIter = 50;
  double tolerance = 1.0e-16;
  double deltaLimit = 1.0e4;

  bool converged = isoparameteric_coordinates_for_point_3d(
      basis_,
      elemNodalCoord,
      pointCoord,
      isoParCoord,
      initialGuess,
      maxIter,
      tolerance,
      deltaLimit
  );
  ThrowAssertMsg(parametric_distance_hex(isoParCoord) < 1.0 + 1.0e-6 || !converged,
      "Inconsistency in parametric distance calculation");

  return (converged) ? parametric_distance_hex(isoParCoord) : std::numeric_limits<double>::max();
}

void HigherOrderTetSCS::interpolatePoint(
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

template <int p> void internal_face_grad_op(
  int face_ordinal,
  const AlignedViewType<DoubleType**[3]>& expReferenceGradWeights,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop )
{
  using traits = AlgTraitsQuadPHexPGL<p>;
  const int offset = traits::numFaceIp_ * face_ordinal;
  auto range = std::make_pair(offset, offset + traits::numFaceIp_);
  auto face_weights = Kokkos::subview(expReferenceGradWeights, range, Kokkos::ALL(), Kokkos::ALL());
  generic_grad_op<AlgTraitsHexGL<p>>(face_weights, coords, gradop);
}

#ifndef KOKKOS_ENABLE_CUDA
void HigherOrderTetSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**, DeviceShmem>& coords,
  SharedMemView<DoubleType***, DeviceShmem>& gradop)
{
  switch(nodes1D_ - 1) {
    case 2: return internal_face_grad_op<2>(face_ordinal, expRefGradWeights_, coords, gradop);
    case 3: return internal_face_grad_op<3>(face_ordinal, expRefGradWeights_, coords, gradop);
    case 4: return internal_face_grad_op<4>(face_ordinal, expRefGradWeights_, coords, gradop);
    case USER_POLY_ORDER: return internal_face_grad_op<USER_POLY_ORDER>(face_ordinal, expRefGradWeights_, coords, gradop);
    default: return;
  }
}
#else
void HigherOrderTetSCS::face_grad_op(
  int ,
  SharedMemView<DoubleType**, DeviceShmem>& ,
  SharedMemView<DoubleType***, DeviceShmem>& )
{}
#endif

}  // namespace nalu
} // namespace sierra