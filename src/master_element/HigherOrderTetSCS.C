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

std::vector<double> HigherOrderTetSCS::getCentroid(std::vector<ordinal_type>& nodeOrdinals, std::unique_ptr<ElementDescription>& eleDesc) {
  const double length = (double)nodeOrdinals.size();
  const double factor = 1.0/length;
  std::vector<double> centroid(3, 0.0);
  for (auto nodeOrdinal : nodeOrdinals) {
    for (int i = 0; i < 3; ++i) {        
      const double coord = eleDesc->nodeLocs[nodeOrdinal][i];
      centroid[i] += factor * coord;
    }
  }
  
  return centroid;
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
//        std::cout << "new quadpoint" << std::endl;
        
        int quadIndex = 0;
        if (quadPoint >= quadrature_.num_quad()) {
          quadIndex = quadPoint - quadrature_.num_quad();
        }
        else {
          quadIndex = quadPoint;
        }
        
        // IP weight
        int orientation = left < right ? 1 : -1;
        ipWeights_(countIP) = orientation * quadrature_.weights(quadIndex) * quadrature_.weights(quadIndex);

        // left/right node mapping
        lrscv_(countIP, 0) = left;
        lrscv_(countIP, 1) = right;
//        std::cout << "left node: " << lrscv_(countIP, 0) << ", right node: " << lrscv_(countIP, 1) << std::endl;

        for (int k = 0; k < 2; ++k) { // repeat 2 times because hex shape functions have 8 nodes but the subsurf has 4 nodes
          
          for (int i = 0; i < 4; ++i) { // for each node of the subsurf
            int subsurfaceNodeLocIndex = 24*subElement + 4*subSurf + i;
            
            for (int j = 0; j < 3; ++j) { // for each dimension
              intgLoc_(countIP, j) += (shape_fcnHex_[countHexSF] * subsurfaceNodeLoc_[subsurfaceNodeLocIndex][j]);
            }
            
            countHexSF++;
          }
        }
        
//        std::cout << "isoCalc intgLoc: " << intgLoc_(countIP, 0) << ", " << intgLoc_(countIP, 1) << ", " << intgLoc_(countIP, 2) << std::endl;
        countIP++;
        
      } // ip
    } // subSurf
  } // subElement
}

// TODO adapt to tet, copied from hex
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
  ipWeightsExpFace_ = Kokkos::View<double*>("ip_weightExpFace", numFaceIps);
  
  auto desc = ElementDescription::create(3, polyOrder_, stk::topology::TET_4);
  int numSubfacePerFace;
  std::vector<std::vector<int>> subfaceCreationIndices;
  
  if (polyOrder_ == 1) {
    numSubfacePerFace = 1;
    subfaceCreationIndices = {
      {0, 1, 2}
    };
  }
  else if (polyOrder_ == 2) {
    numSubfacePerFace = 4;
    subfaceCreationIndices = {
      {0, 1, 5},
      {1, 2, 3},
      {5, 1, 3},
      {5, 3, 4}
    };
  }
  else {
    ThrowErrorMsg("Only P1 and P2 is defined for TET_4 elements.");
  }
  
  subsurfaceNodeLocBC_.resize(numFaces * numSubfacePerFace * numSubsurfacesPerSubface_ * 4, std::vector<double>(3));
  
  int countIP = 0;
  
  // iterate through each face of the element
  for (int face = 0; face < numFaces; ++face) {
  
    // iterate through each subface of the face
    for (int subFace = 0; subFace < numSubfacePerFace; ++subFace) {
      
      // compute subsurface node location and save it for later usage in areav computation
      // compute subface centroid
      std::vector<ordinal_type> subfaceOrdinals(3);
      for (int i = 0; i < 3; ++i) {
        const int subfaceNodeIndex = subfaceCreationIndices[subFace][i];
        subfaceOrdinals[i] = desc->faceNodeMap[face][subfaceNodeIndex];
      }
      std::vector<double> subfaceCentroid = getCentroid(subfaceOrdinals, desc);
    
      // compute subedge centroids
      std::vector<std::vector<ordinal_type>> subedgeOrdinals = {
        {subfaceOrdinals[0], subfaceOrdinals[1]},
        {subfaceOrdinals[1], subfaceOrdinals[2]},
        {subfaceOrdinals[2], subfaceOrdinals[0]}
      };
      std::vector<double> subedge1Centroid = getCentroid(subedgeOrdinals[0], desc);
      std::vector<double> subedge2Centroid = getCentroid(subedgeOrdinals[1], desc);
      std::vector<double> subedge3Centroid = getCentroid(subedgeOrdinals[2], desc);
        
      const int subsurfaceNodeLocIndex = face*numSubfacePerFace*numSubsurfacesPerSubface_*4 + subFace*numSubsurfacesPerSubface_*4;
//      std::cout << "subsurfaceNodeLocIndex = " << subsurfaceNodeLocIndex << std::endl;
      
      // subsurface 1
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 0] = subfaceCentroid;
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 1] = subedge1Centroid;
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 2] = desc->nodeLocs[subfaceOrdinals[0]];
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 3] = subedge3Centroid;
      
      // subsurface 2
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 4] = subfaceCentroid;
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 5] = subedge2Centroid;
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 6] = desc->nodeLocs[subfaceOrdinals[1]];
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 7] = subedge1Centroid;
      
      // subsurface 3
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 8] = subfaceCentroid;
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 9] = subedge3Centroid;
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 10] = desc->nodeLocs[subfaceOrdinals[2]];
      subsurfaceNodeLocBC_[subsurfaceNodeLocIndex + 11] = subedge2Centroid;
      
      // isoparametric mapping
      for (int subSurf = 0; subSurf < 3; ++subSurf) {
        
        const int nearNode = subfaceOrdinals[subSurf];
        
        // isoparametric mapping of the intgLoc of a isoparametric rectangle to the isoparametric tet
        int countHexSF = 0;
        for (int quadPoint = 0; quadPoint < numQuad_; ++quadPoint) { // for each ip at subsurf
//          std::cout << "new quadpoint" << std::endl;
          
          int quadIndex = 0;
          if (quadPoint >= quadrature_.num_quad()) {
            quadIndex = quadPoint - quadrature_.num_quad();
          }
          else {
            quadIndex = quadPoint;
          }

          // IP weight
          int orientation = 1;
          ipWeightsExpFace_(countIP) = orientation * quadrature_.weights(quadIndex) * quadrature_.weights(quadIndex);
          ipNodeMap_(countIP) = nearNode;

          for (int k = 0; k < 2; ++k) { // repeat 2 times because hex shape functions have 8 nodes but the subsurf has 4 nodes

            for (int i = 0; i < 4; ++i) { // for each node of the subsurf
              const int subsurfaceNodeLocIndex = face*numSubfacePerFace*numSubsurfacesPerSubface_*4 + subFace*numSubsurfacesPerSubface_*4 + subSurf*4 + i;
//              std::cout << "second subsurfaceNodeLocIndex = " << subsurfaceNodeLocIndex << std::endl;

              for (int j = 0; j < 3; ++j) { // for each dimension
                intgExpFace_(countIP, j) += (shape_fcnHex_[countHexSF] * subsurfaceNodeLocBC_[subsurfaceNodeLocIndex][j]);
              }

              countHexSF++;
            }
          }

//          std::cout << "isoCalc intgExpFace: " << intgExpFace_(countIP, 0) << ", " << intgExpFace_(countIP, 1) << ", " << intgExpFace_(countIP, 2) << std::endl;
          countIP++;

        } // ip
      }
    }
  }
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
HigherOrderTetSCS::shape_fcn(double* shpfc)
{
  
  if (polyOrder_ == 1) {
    tet_shape_fcn_p1(numIntPoints_, intgLoc_, shpfc);
  }
  else if (polyOrder_ == 2) {
    tet_shape_fcn_p2(numIntPoints_, intgLoc_, shpfc);
  }
  else {
    ThrowErrorMsg("Shape functions not defined for the chosen polyOrder");
  }
}

void HigherOrderTetSCS::tet_shape_fcn_p1(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int fourj = 4*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    const double zeta = par_coord(j, 2);
    shape_fcn[0 + fourj] = 1.0 - xi - eta - zeta;
    shape_fcn[1 + fourj] = xi;
    shape_fcn[2 + fourj] = eta;
    shape_fcn[3 + fourj] = zeta;
  }
}

void HigherOrderTetSCS::tet_shape_fcn_p2(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int tenj = 10*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    const double zeta = par_coord(j, 2);
    
    const double L1 = 1-xi-eta-zeta;
    const double L2 = xi;
    const double L3 = eta;
    const double L4 = zeta;
    
    shape_fcn[0 + tenj] = L1*(2*L1-1);
    shape_fcn[1 + tenj] = L2*(2*L2-1);
    shape_fcn[2 + tenj] = L3*(2*L3-1);
    shape_fcn[3 + tenj] = L4*(2*L4-1);
    shape_fcn[4 + tenj] = 4*L1*L2;
    shape_fcn[5 + tenj] = 4*L2*L3;
    shape_fcn[6 + tenj] = 4*L3*L1;
    shape_fcn[7 + tenj] = 4*L1*L4;
    shape_fcn[8 + tenj] = 4*L2*L4;
    shape_fcn[9 + tenj] = 4*L3*L4;
  }
}

void HigherOrderTetSCS::tet_deriv_shape_fcn_p1(
  const int   npts, 
  double *deriv)
{
  for (int j = 0; j < npts; ++j ) {
    const int twelvej = 12*j;
    deriv[0  + twelvej] = -1.0;   // IP j, Node 0, dxi
    deriv[1  + twelvej] = -1.0;   // IP j, Node 0, deta
    deriv[2  + twelvej] = -1.0;   // IP j, Node 0, dzeta
    deriv[3  + twelvej] =  1.0;   // IP j, Node 1, dxi
    deriv[4  + twelvej] =  0.0;   // IP j, Node 1, deta
    deriv[5  + twelvej] =  0.0;   // IP j, Node 1, dzeta
    deriv[6  + twelvej] =  0.0;   // IP j, Node 2, dxi
    deriv[7  + twelvej] =  1.0;   // IP j, Node 2, deta
    deriv[8  + twelvej] =  0.0;   // IP j, Node 2, dzeta
    deriv[9  + twelvej] =  0.0;   // IP j, Node 3, dxi
    deriv[10 + twelvej] =  0.0;   // IP j, Node 3, deta
    deriv[11 + twelvej] =  1.0;   // IP j, Node 3, dzeta
  }
}

void HigherOrderTetSCS::tet_deriv_shape_fcn_p2(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *deriv)
{
  for (int j = 0; j < npts; ++j ) {
    const int thirtyj = 30*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    const double zeta = par_coord(j, 2);
    deriv[0  + thirtyj] =  4.0*xi+4.0*eta+4.0*zeta-3;   // IP j, Node 0, dxi
    deriv[1  + thirtyj] =  4.0*xi+4.0*eta+4.0*zeta-3;   // IP j, Node 0, deta
    deriv[2  + thirtyj] =  4.0*xi+4.0*eta+4.0*zeta-3;   // IP j, Node 0, dzeta
    deriv[3  + thirtyj] =  4.0*xi-1;                    // IP j, Node 1, dxi
    deriv[4  + thirtyj] =  0.0;                         // IP j, Node 1, deta
    deriv[5  + thirtyj] =  0.0;                         // IP j, Node 1, dzeta
    deriv[6  + thirtyj] =  0.0;                         // IP j, Node 2, dxi
    deriv[7  + thirtyj] =  4.0*eta-1;                   // IP j, Node 2, deta
    deriv[8  + thirtyj] =  0.0;                         // IP j, Node 2, dzeta
    deriv[9  + thirtyj] =  0.0;                         // IP j, Node 3, dxi
    deriv[10 + thirtyj] =  0.0;                         // IP j, Node 3, deta
    deriv[11 + thirtyj] =  4.0*zeta-1;                  // IP j, Node 3, dzeta
    deriv[12 + thirtyj] = -4.0*(2.0*xi+eta+zeta-1);     // IP j, Node 4, dxi
    deriv[13 + thirtyj] = -4.0*xi;                      // IP j, Node 4, deta
    deriv[14 + thirtyj] = -4.0*xi;                      // IP j, Node 4, dzeta
    deriv[15 + thirtyj] =  4.0*eta;                     // IP j, Node 5, dxi
    deriv[16 + thirtyj] =  4.0*xi;                      // IP j, Node 5, deta
    deriv[17 + thirtyj] =  0.0;                         // IP j, Node 5, dzeta
    deriv[18 + thirtyj] = -4.0*eta;                     // IP j, Node 6, dxi
    deriv[19 + thirtyj] = -4.0*(xi+2.0*eta+zeta-1);     // IP j, Node 6, deta
    deriv[20 + thirtyj] = -4.0*eta;                     // IP j, Node 6, dzeta
    deriv[21 + thirtyj] = -4.0*zeta;                    // IP j, Node 7, dxi
    deriv[22 + thirtyj] = -4.0*zeta;                    // IP j, Node 7, deta
    deriv[23 + thirtyj] = -4.0*(xi+eta+2.0*zeta-1);     // IP j, Node 7, dzeta
    deriv[24 + thirtyj] =  4.0*zeta;                    // IP j, Node 8, dxi
    deriv[25 + thirtyj] =  0.0;                         // IP j, Node 8, deta
    deriv[26 + thirtyj] =  4.0*xi;                      // IP j, Node 8, dzeta
    deriv[27 + thirtyj] =  0.0;                         // IP j, Node 9, dxi
    deriv[28 + thirtyj] =  4.0*zeta;                    // IP j, Node 9, deta
    deriv[29 + thirtyj] =  4.0*eta;                     // IP j, Node 9, dzeta
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
  const int numSubSurf = numIntPoints_ / numQuad_;
  Kokkos::View<double**> realCoords;
  Kokkos::View<double**> isoParCoords;
  realCoords = Kokkos::View<double[4][3]>("realCoords");
  isoParCoords = Kokkos::View<double[4][3]>("isoParCoords");
  std::vector<double> shape_fcn(4 * nodesPerElement_);
  double *p_shape_fcn = &shape_fcn[0];

  // loop through all internal subsurfaces (scs)
  int offset = 0;
  for (int subSurf = 0; subSurf < numSubSurf; ++subSurf) {
    
    // initialize coords vectors
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 3; ++j) {
        realCoords(i, j) = 0.0;
        isoParCoords(i, j) = subsurfaceNodeLoc_[subSurf*4 + i][j];
      }
    }
    
    // evaluate shape functions at the vertices of the scs
    if (polyOrder_ == 1) {
      tet_shape_fcn_p1(4, isoParCoords, &p_shape_fcn[0]);
    }
    else if (polyOrder_ == 2) {
      tet_shape_fcn_p2(4, isoParCoords, &p_shape_fcn[0]);
    }
    else {
    ThrowErrorMsg("Shape functions not defined for the chosen polyOrder");
    }
    
    // use isoparametric mapping to get real coordinates of the vertices
    int count = 0;
    for (int vert = 0; vert < 4; ++vert) {
      for (int node = 0; node < nodesPerElement_; ++node) {
        for (int j = 0; j < 3; ++j) {
          realCoords(vert, j) += (shape_fcn[count] * coords[node * nDim_ + j]);
        }
        count++;
      }
    }
    
    // calculate area vector = 1/2 * cross product of the diagonals
    std::vector<double> d1 = {realCoords(2,0)-realCoords(0,0), 
                              realCoords(2,1)-realCoords(0,1), 
                              realCoords(2,2)-realCoords(0,2)};
    
    std::vector<double> d2 = {realCoords(3,0)-realCoords(1,0), 
                              realCoords(3,1)-realCoords(1,1), 
                              realCoords(3,2)-realCoords(1,2)};
    
    std::vector<double> area_vector = {0.5*(d1[1]*d2[2]-d1[2]*d2[1]),
                                       0.5*(d1[2]*d2[0]-d1[0]*d2[2]),
                                       0.5*(d1[0]*d2[1]-d1[1]*d2[0])};
    
    // loop through all IPs of the current scs
    for (int ip = 0; ip < numQuad_; ++ip) {
      const double weight = ipWeights_(offset + ip);
      for (int j = 0; j < 3; ++j) {
        areav[(offset + ip) * nDim_ + j] = area_vector[j] * weight;
      }
    }
    
    offset += numQuad_;
  }
  *error = 0; // no error checking available
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
  
  if (polyOrder_ == 1) {
    tet_deriv_shape_fcn_p1(numIntPoints_, deriv);
  }
  else if (polyOrder_ == 2) {
    tet_deriv_shape_fcn_p2(numIntPoints_, intgLoc_, deriv);
  }
  else {
    ThrowErrorMsg("Shape function derivatives not defined for the chosen polyOrder");
  }

  for (int ip = 0; ip < numIntPoints_; ++ip) {

    gradient_3d(
      nodesPerElement_,
      coords,
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

// TODO adapt to tet, copied from hex
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

// TODO adapt to tet, copied from hex
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

// TODO adapt to tet, copied from hex
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

// TODO adapt to tet, copied from hex
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