/* 
 * File:   HigherOrderTri3DSCS.C
 * Author: Raphael Lindegger
 * 
 * Created on November 2, 2019, 12:59 PM
 */

#include <master_element/HigherOrderTri3DSCS.h>
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
#include <limits>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace sierra{
namespace nalu{

KOKKOS_FUNCTION
HigherOrderTri3DSCS::HigherOrderTri3DSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature)),
#ifndef KOKKOS_ENABLE_CUDA
  nodeMap(make_node_map_tri(basis.order())),
#endif
  nodes1D_(basis.order()+1),
  numQuad_(quadrature.num_quad()*quadrature.num_quad()),
  numSubsurfacesPerSubelement_(3),
  polyOrder_(nodes1D_-1)
{
  surfaceDimension_ = 2;
  nDim_ = 3;
  nodesPerElement_ = 0.5*(nodes1D_*(nodes1D_+1)); // triangular number
  
    // generate hex shape functions used for the isoparametric mapping intgLoc on subsurfaces (scs)
  intgLocSurfIso_ = Kokkos::View<double**>("integration_point_location_subsurf", numQuad_, 2);
  if (polyOrder_ == 1) {
    // define IP location in isoparametric subsurface
    // IP1, there is just one for P1
    intgLocSurfIso_(0, 0) = quadrature_.abscissa(0);
    intgLocSurfIso_(0, 1) = quadrature_.abscissa(0); 
  }
  else if (polyOrder_ == 2) {
    // define IP locations in isoparametric subsurface
    // IP1
    intgLocSurfIso_(0, 0) = quadrature_.abscissa(0);
    intgLocSurfIso_(0, 1) = quadrature_.abscissa(0);
    // IP2
    intgLocSurfIso_(1, 0) = quadrature_.abscissa(1);
    intgLocSurfIso_(1, 1) = quadrature_.abscissa(0);
    // IP3
    intgLocSurfIso_(2, 0) = quadrature_.abscissa(0);
    intgLocSurfIso_(2, 1) = quadrature_.abscissa(1);
    // IP4
    intgLocSurfIso_(3, 0) = quadrature_.abscissa(1);
    intgLocSurfIso_(3, 1) = quadrature_.abscissa(1);
  }
  else {
    ThrowErrorMsg("Only P1 and P2 is defined for TET_4 elements.");
  }
  
  shape_fcnQuad_.resize(numQuad_ * 4);
  double *p_shape_fcnQuad = &shape_fcnQuad_[0];
  quad_shape_fcn_p1(numQuad_, intgLocSurfIso_, &p_shape_fcnQuad[0]);

#ifndef KOKKOS_ENABLE_CUDA
  // set up integration rule and relevant maps on scs
  set_interior_info();
#endif
}

std::vector<double> 
HigherOrderTri3DSCS::getCentroid(std::vector<ordinal_type>& nodeOrdinals, std::unique_ptr<ElementDescription>& eleDesc) {
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

void
HigherOrderTri3DSCS::set_interior_info()
{
  auto desc = ElementDescription::create(2, polyOrder_, stk::topology::TRI_3_2D);
  
  if (polyOrder_ == 1) 
    numSubelements_ = 1;
  else if (polyOrder_ == 2)
    numSubelements_ = 4;
  else
    ThrowErrorMsg("Only P1 and P2 is defined for TET_4 elements.");

  // define ip node mappings
  numIntPoints_ = numSubelements_ * numSubsurfacesPerSubelement_ * numQuad_;
  ipNodeMap_= Kokkos::View<int*>("ipNodeMap_", numIntPoints_);
  intgLoc_ = Kokkos::View<double**>("integration_point_location", numIntPoints_, 2);
  ipWeights_ = Kokkos::View<double*>("ip_weight", numIntPoints_);
  subsurfaceNodeLoc_.resize(numSubelements_ * numSubsurfacesPerSubelement_ * 4, std::vector<double>(2));
  ordinal_type subcontrol_vol_ord;
  
  int countIP = 0;
  
  std::vector<std::vector<int>> subsurfCreationIndices {
    {0, 1, 2}, // subelement centroid
    {0, 2},    // subedge 1 centroid
    {0},       // node
    {0, 1},    // subedge 3 centroid
    {0, 1, 2}, // subelement centroid
    {0, 1},    // subedge 1 centroid
    {1},       // node
    {1, 2},    // subedge 3 centroid
    {0, 1, 2}, // subelement centroid
    {1, 2},    // subedge 1 centroid
    {2},       // node
    {0, 2},    // subedge 3 centroid
  };

  // initialize intgLoc_
  for (int i = 0; i < numIntPoints_; ++i) {
    for (int j = 0; j < surfaceDimension_; ++j) {
      intgLoc_(i, j) = 0.0;
    }
  }
   
  // loop through each subelement and compute the integration points at each subsurface in the subelement
  int countNode = 0;
  for (int subElement = 0; subElement < numSubelements_; ++subElement) {
    
    int countSubsurf = 0;
    for (int subSurf = 0; subSurf < 3; ++subSurf) {
      
      for (int node = 0; node < 4; ++node) {
        const int numOrd = subsurfCreationIndices[countSubsurf].size();
        std::vector<ordinal_type> centroidDefiningOrdinals(numOrd);
        
        for (int i = 0; i < numOrd; ++i) {
          const int ordIndex = subsurfCreationIndices[countSubsurf][i];
          centroidDefiningOrdinals[i] = desc->subElementConnectivity[subElement][ordIndex];
        }
        
        if (numOrd == 1) {
          subcontrol_vol_ord = centroidDefiningOrdinals[0];
        }
        
        // compute subsurface node location and save it for later usage in areav computation
        std::vector<double> centroid = getCentroid(centroidDefiningOrdinals, desc);
        std::vector<double> nodeLoc(2, 0.0);
        nodeLoc[0] = centroid[0];
        nodeLoc[1] = centroid[1];
        
        int subsurfaceNodeLocIndex = 12*subElement + 4*subSurf + node;
        subsurfaceNodeLoc_[subsurfaceNodeLocIndex] = nodeLoc;
        
        countSubsurf++;
      }

      // isoparametric mapping of the intgLoc of a isoparametric rectangle to the isoparametric tri
      int countQuadSF = 0;
      int quadIndex = 0;
      for (int quadPoint = 0; quadPoint < numQuad_; ++quadPoint) { // for each ip at subsurf
//        std::cout << "new quadpoint" << std::endl;
        
        //sub-control volume association
        ipNodeMap_(countIP) = subcontrol_vol_ord;

        if (quadIndex >= quadrature_.num_quad()) {
          quadIndex = 0;
        }

        // IP weight
        int orientation = 1;
        ipWeights_(countIP) = orientation * quadrature_.weights(quadIndex) * quadrature_.weights(quadIndex);
          
          // isoparametric mapping
          for (int i = 0; i < 4; ++i) { // for each node of the subsurf
            int subsurfaceNodeLocIndex = 12*subElement + 4*subSurf + i;
            
            for (int j = 0; j < 2; ++j) { // for each dimension
              intgLoc_(countIP, j) += (shape_fcnQuad_[countQuadSF] * subsurfaceNodeLoc_[subsurfaceNodeLocIndex][j]);
            }
            
            countQuadSF++;
          }
        
//        std::cout << "isoCalc intgLoc: " << intgLoc_(countIP, 0) << ", " << intgLoc_(countIP, 1) << std::endl;
        countIP++;
        quadIndex++;
        
      } // ip
    } // subSurf
  } // subElement
}

void
HigherOrderTri3DSCS::quad_shape_fcn_p1(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int fourj = 4*j;
    const double oneHalf = 1.0/2.0;
    const double xi   = par_coord(j, 0);
    const double eta  = par_coord(j, 1);
    shape_fcn[0 + fourj] = oneHalf*(1.0-xi)*oneHalf*(1.0-eta);
    shape_fcn[1 + fourj] = oneHalf*(1.0+xi)*oneHalf*(1.0-eta);
    shape_fcn[2 + fourj] = oneHalf*(1.0+xi)*oneHalf*(1.0+eta);
    shape_fcn[3 + fourj] = oneHalf*(1.0-xi)*oneHalf*(1.0+eta);
  }
}

void
HigherOrderTri3DSCS::shape_fcn(double* shpfc)
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

void HigherOrderTri3DSCS::tri_shape_fcn_p1(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int threej = 3*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    shape_fcn[0 + threej] = 1.0 - xi - eta;
    shape_fcn[1 + threej] = xi;
    shape_fcn[2 + threej] = eta;
  }
}

void HigherOrderTri3DSCS::tri_shape_fcn_p2(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int sixj = 6*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    shape_fcn[0 + sixj] = (1.0-xi-eta)*(1.0-2.0*xi-2.0*eta);
    shape_fcn[1 + sixj] = xi*(2.0*xi-1.0);
    shape_fcn[2 + sixj] = eta*(2.0*eta-1.0);
    shape_fcn[3 + sixj] = 4.0*xi*(1.0-xi-eta);
    shape_fcn[4 + sixj] = 4.0*xi*eta;
    shape_fcn[5 + sixj] = 4.0*eta*(1.0-xi-eta);
  }
}

void HigherOrderTri3DSCS::tri_deriv_shape_fcn_p1(
  const int   npts, 
  double *deriv)
{
  for (int j = 0; j < npts; ++j ) {
    const int sixj = 6*j;
    deriv[0  + sixj] = -1.0;   // IP j, Node 0, dxi
    deriv[1  + sixj] = -1.0;   // IP j, Node 0, deta
    deriv[2  + sixj] =  1.0;   // IP j, Node 1, dxi
    deriv[3  + sixj] =  0.0;   // IP j, Node 1, deta
    deriv[4  + sixj] =  0.0;   // IP j, Node 2, dxi
    deriv[5  + sixj] =  1.0;   // IP j, Node 2, deta
  }
}

void HigherOrderTri3DSCS::tri_deriv_shape_fcn_p2(
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
    deriv[11 + twelvej] = -4.0*(2.0*eta+xi-1.0);     // IP j, Node 5, deta
  }
}

void HigherOrderTri3DSCS::pri_shape_fcn_p1(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int sixj = 6*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    const double zeta = par_coord(j, 2);
    shape_fcn[0 + sixj] = 0.5*(1.0-xi-eta)*(1.0-zeta);
    shape_fcn[1 + sixj] = 0.5*xi*(1.0-zeta);
    shape_fcn[2 + sixj] = 0.5*eta*(1.0-zeta);
    shape_fcn[3 + sixj] = 0.5*(1.0-xi-eta)*(1.0+zeta);
    shape_fcn[4 + sixj] = 0.5*xi*(1.0+zeta);
    shape_fcn[5 + sixj] = 0.5*eta*(1.0+zeta);
  }
}

void HigherOrderTri3DSCS::pri_shape_fcn_p2(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int twelvej = 6*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    const double zeta = par_coord(j, 2);
    shape_fcn[0  + twelvej] = 0.5*(1.0-xi-eta)*(1.0-zeta);
    shape_fcn[1  + twelvej] = 0.5*xi*(1.0-zeta);
    shape_fcn[2  + twelvej] = 0.5*eta*(1.0-zeta);
    shape_fcn[3  + twelvej] = 0.5*(1.0-xi-eta)*(1.0+zeta);
    shape_fcn[4  + twelvej] = 0.5*xi*(1.0+zeta);
    shape_fcn[5  + twelvej] = 0.5*eta*(1.0+zeta);
    shape_fcn[6  + twelvej] = 2.0*xi*(1.0-xi-eta)*(1.0-zeta);
    shape_fcn[7  + twelvej] = 2.0*xi*eta*(1.0-zeta);
    shape_fcn[8  + twelvej] = 2.0*eta*(1.0-xi-eta)*(1.0-zeta);
    shape_fcn[9  + twelvej] = 2.0*xi*(1.0-xi-eta)*(1.0+zeta);
    shape_fcn[10 + twelvej] = 2.0*xi*eta*(1.0+zeta);
    shape_fcn[11 + twelvej] = 2.0*eta*(1.0-xi-eta)*(1.0+zeta);
  }
}

const int *
HigherOrderTri3DSCS::ipNodeMap(
  int /*ordinal*/) const
{
  // define ip->node mappings for each face (single ordinal);
  return &ipNodeMap_(0);
}

void
HigherOrderTri3DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double * /*error*/)
{
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

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
      pri_shape_fcn_p1(4, isoParCoords, &p_shape_fcn[0]);
    }
    else if (polyOrder_ == 2) {
      pri_shape_fcn_p2(4, isoParCoords, &p_shape_fcn[0]);
    }
    else {
    ThrowErrorMsg("Shape functions not defined for the chosen polyOrder");
    }
    
    // use isoparametric mapping to get real coordinates of the vertices
    int count = 0;
    for (int vert = 0; vert < 4; ++vert) {
      for (int k = 0; k < 2; ++k) { // repeat 2 times because pri shape functions have twice as many nodes than the element
        for (int node = 0; node < nodesPerElement_; ++node) {
          for (int j = 0; j < 3; ++j) {
            realCoords(vert, j) += (shape_fcn[count] * coords[node * nDim_ + j]);
          }
          count++;
        }
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
    
//    std::cout << "area_vector = {" << area_vector[0] << ", " << area_vector[1] << ", " << area_vector[2] << "}" << std::endl;
    // loop through all IPs of the current scs
    for (int ip = 0; ip < numQuad_; ++ip) {
      const double weight = ipWeights_(offset + ip);
      for (int j = 0; j < 3; ++j) {
        areav[(offset + ip) * nDim_ + j] = area_vector[j] * weight;
      }
    }
    
    offset += numQuad_;
  }
}

}  // namespace nalu
} // namespace sierra