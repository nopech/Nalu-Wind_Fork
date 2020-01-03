/* 
 * File:   HigherOrderTri2DSCV.C
 * Author: Raphael Lindegger
 * 
 * Created on October 20, 2019, 10:42 AM
 */

#include <master_element/HigherOrderTri2DSCV.h>
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

namespace sierra{
namespace nalu{
  
KOKKOS_FUNCTION
HigherOrderTri2DSCV::HigherOrderTri2DSCV(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(std::move(basis)),
  totalVol_(0.0),
  quadrature_(std::move(quadrature)),
  numQuad_(quadrature.num_quad()*quadrature.num_quad()),
#ifndef KOKKOS_ENABLE_CUDA
  nodeMap(make_node_map_tri(basis.order())),
#endif
  nodes1D_(basis.order()+1),
  polyOrder_(nodes1D_-1)
{
  MasterElement::nDim_ = 2;
  nodesPerElement_ = 0.5*(nodes1D_*(nodes1D_+1));

#ifndef KOKKOS_ENABLE_CUDA
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
  
  
  // set up integration rule and relevant maps for scv's
  set_interior_info();
#endif
}

std::vector<double> 
HigherOrderTri2DSCV::getCentroid(std::vector<ordinal_type> nodeOrdinals, std::unique_ptr<ElementDescription>& eleDesc) {
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
HigherOrderTri2DSCV::set_interior_info()
{
  auto desc = ElementDescription::create(2, polyOrder_, stk::topology::TRI_3_2D);
  
//  std::cout << std::endl;
//  std::cout << "set SCV interior info" << std::endl;
  
  int IPCount = 0;
  int subVolCount = 0;
  ordinal_type nodeOrdinal;
  std::vector<double> subTriCentroid(2);
  std::vector<double> bottomEdgeCentroid(2);
  std::vector<double> rightEdgeCentroid(2);
  std::vector<double> leftEdgeCentroid(2);
  std::vector<double> nodeCoord(2);

  // lambda to compute the integration location and weight
  auto writeIPInfo = [&](int& subVol, int& count, std::vector<double> centroidOne, std::vector<double> centroidTwo, std::vector<double> centroidThree, std::vector<double> centroidFour, ordinal_type ord) {
    
    // Save BP vector (subface) for later usage in areav computation
    // Note that the coords are in the isoparametric coord frame
    intSubVolumes_(subVol, 0, 0) = centroidOne[0];
    intSubVolumes_(subVol, 0, 1) = centroidOne[1];
    intSubVolumes_(subVol, 1, 0) = centroidTwo[0];
    intSubVolumes_(subVol, 1, 1) = centroidTwo[1];
    intSubVolumes_(subVol, 2, 0) = centroidThree[0];
    intSubVolumes_(subVol, 2, 1) = centroidThree[1];
    intSubVolumes_(subVol, 3, 0) = centroidFour[0];
    intSubVolumes_(subVol, 3, 1) = centroidFour[1];
    
    int countQuadSF = 0;
    int quadIndex = 0;
    for (int j = 0; j < numQuad_; ++j) {
      if (quadIndex >= quadrature_.num_quad()) {
        quadIndex = 0;
      }
      
      ipNodeMap_(count) = ord;
      ipWeights_(count) = quadrature_.weights(quadIndex) * quadrature_.weights(quadIndex);
      
      // isoparametric mapping from iso quad to scv quad
      for (int i = 0; i < 4; ++i) { // for each node of the scv
        for (int j = 0; j < 2; ++j) { // for each dimension
          intgLoc_(count, j) += (shape_fcnQuad_[countQuadSF] * intSubVolumes_(subVol, i, j));
        }
        countQuadSF++;
      }
      
//      std::cout << "ipWeights_(" << count << ") = " << ipWeights_(count) << std::endl;
//      std::cout << "intgLoc_(" << count << ") = {" << intgLoc_(count, 0) << ", " << intgLoc_(count, 1) << "}" << std::endl;
      
      count++;
      quadIndex++;
    }
    
    subVol++;
  };

  // Hardcode left/right node mapping and integration locations
  if (polyOrder_ == 1) {
    numIntPoints_ = 3;

    // standard integration location
    ipNodeMap_= Kokkos::View<int*>("ipNodeMap_", numIntPoints_);
    intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
    ipWeights_ = Kokkos::View<double*>("ipWeights_", numIntPoints_);
    intSubVolumes_ = Kokkos::View<double*[4][2]>("intSubfaces_", numIntPoints_/numQuad_);
    
    subTriCentroid = getCentroid({0, 1, 2}, desc);
    bottomEdgeCentroid = getCentroid({0, 1}, desc);
    rightEdgeCentroid = getCentroid({1, 2}, desc);
    leftEdgeCentroid = getCentroid({2, 0}, desc);
    
    // IP for node 0
    nodeOrdinal = 0;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
    
    // IP for node 1
    nodeOrdinal = 1;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, rightEdgeCentroid, nodeOrdinal);
    
    // IP for node 2
    nodeOrdinal = 2;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, rightEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
  }
  else if (polyOrder_ == 2) {
    numIntPoints_ = 48;

    // standard integration location
    ipNodeMap_= Kokkos::View<int*>("ipNodeMap_", numIntPoints_);
    intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
    ipWeights_ = Kokkos::View<double*>("ipWeights_", numIntPoints_);
    intSubVolumes_ = Kokkos::View<double*[4][2]>("intSubfaces_", numIntPoints_/numQuad_);
    
    //----------------------------------------------------------------
    // bottom left subtriangle
    subTriCentroid = getCentroid({0, 3, 5}, desc);
    bottomEdgeCentroid = getCentroid({0, 3}, desc);
    rightEdgeCentroid = getCentroid({3, 5}, desc);
    leftEdgeCentroid = getCentroid({5, 0}, desc);
    
    // IP for node 0
    nodeOrdinal = 0;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
    
    // IP for node 3
    nodeOrdinal = 3;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, rightEdgeCentroid, nodeOrdinal);
    
    // IP for node 5
    nodeOrdinal = 5;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, rightEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
    
    //----------------------------------------------------------------
    // bottom right subtriangle
    subTriCentroid = getCentroid({3, 1, 4}, desc);
    bottomEdgeCentroid = getCentroid({3, 1}, desc);
    rightEdgeCentroid = getCentroid({1, 4}, desc);
    leftEdgeCentroid = getCentroid({4, 3}, desc);
    
    // IP for node 3
    nodeOrdinal = 3;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
    
    // IP for node 1
    nodeOrdinal = 1;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, rightEdgeCentroid, nodeOrdinal);
    
    // IP for node 4
    nodeOrdinal = 4;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, rightEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
    
    //----------------------------------------------------------------
    // top subtriangle
    subTriCentroid = getCentroid({5, 4, 2}, desc);
    bottomEdgeCentroid = getCentroid({5, 4}, desc);
    rightEdgeCentroid = getCentroid({4, 2}, desc);
    leftEdgeCentroid = getCentroid({2, 5}, desc);
    
    // IP for node 5
    nodeOrdinal = 5;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
    
    // IP for node 4
    nodeOrdinal = 4;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, rightEdgeCentroid, nodeOrdinal);
    
    // IP for node 2
    nodeOrdinal = 2;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, rightEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
    
    //----------------------------------------------------------------
    // middle subtriangle
    subTriCentroid = getCentroid({3, 4, 5}, desc);
    bottomEdgeCentroid = getCentroid({3, 4}, desc);
    rightEdgeCentroid = getCentroid({4, 5}, desc);
    leftEdgeCentroid = getCentroid({5, 3}, desc);
    
    // IP for node 3
    nodeOrdinal = 3;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
    
    // IP for node 4
    nodeOrdinal = 4;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, bottomEdgeCentroid, nodeCoord, rightEdgeCentroid, nodeOrdinal);
    
    // IP for node 5
    nodeOrdinal = 5;
    nodeCoord = {desc->nodeLocs[nodeOrdinal][0], desc->nodeLocs[nodeOrdinal][1]};
    writeIPInfo(subVolCount, IPCount, subTriCentroid, rightEdgeCentroid, nodeCoord, leftEdgeCentroid, nodeOrdinal);
  }
  else {
    ThrowErrorMsg("Only P1 and P2 is defined for TRI_2_2D elements.");
  }
}

void
HigherOrderTri2DSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");
  const int ipsPerSubvol = numQuad_;
  const int numSubVol = numIntPoints_/numQuad_;
  Kokkos::View<double**> realCoords;
  Kokkos::View<double**> isoParCoords;

  // Loop through all internal faces
  int countIP = 0;
  for (int vol = 0; vol < numSubVol; ++vol) {
//    std::cout << "face: " << face << std::endl;
    realCoords = Kokkos::View<double[4][2]>("realCoords");
    isoParCoords = Kokkos::View<double[4][2]>("isoParCoords");
    realCoords(0, 0) = 0.0;
    realCoords(0, 1) = 0.0;
    realCoords(1, 0) = 0.0;
    realCoords(1, 1) = 0.0;
    realCoords(2, 0) = 0.0;
    realCoords(2, 1) = 0.0;
    realCoords(3, 0) = 0.0;
    realCoords(3, 1) = 0.0;
    isoParCoords(0, 0) = intSubVolumes_(vol, 0, 0);
    isoParCoords(0, 1) = intSubVolumes_(vol, 0, 1);
    isoParCoords(1, 0) = intSubVolumes_(vol, 1, 0);
    isoParCoords(1, 1) = intSubVolumes_(vol, 1, 1);
    isoParCoords(2, 0) = intSubVolumes_(vol, 2, 0);
    isoParCoords(2, 1) = intSubVolumes_(vol, 2, 1);
    isoParCoords(3, 0) = intSubVolumes_(vol, 3, 0);
    isoParCoords(3, 1) = intSubVolumes_(vol, 3, 1);
    
    std::vector<double> shape_fcn(4 * nodesPerElement_);
    double *p_shape_fcn = &shape_fcn[0];

    // Evaluate shape functions at the vertices of the scv
    if (polyOrder_ == 1) {
      tri_shape_fcn_p1(4, isoParCoords, &p_shape_fcn[0]);
    }
    else if (polyOrder_ == 2) {
      tri_shape_fcn_p2(4, isoParCoords, &p_shape_fcn[0]);
    }
    else {
    ThrowErrorMsg("Shape functions not defined for the chosen polyOrder");
    }
    
    // Use isoparametric mapping to get real coordinates of the vertices of the scv
    int count = 0;
    for (int vert = 0; vert < 4; ++vert) {
      for (int node = 0; node < nodesPerElement_; ++node) {
        for (int j = 0; j < 2; ++j) {
          realCoords(vert, j) += (shape_fcn[count] * coords[node * nDim_ + j]);;
        }
        count++;
      }
    }
    
    const double x1 = realCoords(0, 0); const double y1 = realCoords(0, 1);
    const double x2 = realCoords(1, 0); const double y2 = realCoords(1, 1);
    const double x3 = realCoords(2, 0); const double y3 = realCoords(2, 1);
    const double x4 = realCoords(3, 0); const double y4 = realCoords(3, 1);
    
    const double vol_scv = std::abs(0.5*((x1*y2-y1*x2) + (x2*y3-y2*x3) + (x3*y4-y3*x4) + (x4*y1-y4*x1)));

//    std::cout << "vol_scv = " << vol_scv << std::endl;
//    totalVol_ += vol_scv;

    // Loop through all IPs of the current scv
    for (int quadPoint = 0; quadPoint < numQuad_; ++quadPoint) {
      volume[countIP] = ipWeights_(countIP) * vol_scv;
//      std::cout << "ipWeight = " << ipWeights_(countIP) << ", " << "vol_scv = " << vol_scv << std::endl;
      countIP++;
    }
  }
  
//  std::cout << "total volume = " << totalVol_ << std::endl;
}

void
HigherOrderTri2DSCV::shape_fcn(double *shpfc)
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

const int *
HigherOrderTri2DSCV::ipNodeMap(int /*ordinal*/) const
{
  return &ipNodeMap_(0);
}

void
HigherOrderTri2DSCV::quad_shape_fcn_p1(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int fourj = 4*j;
    const double oneHalf = 1.0/2.0;
    const double xi   = par_coord(j, 0);
    const double eta  = par_coord(j, 1);
    shape_fcn[0 + fourj] = oneHalf*(1.0-xi) * oneHalf*(1.0-eta);
    shape_fcn[1 + fourj] = oneHalf*(1.0+xi) * oneHalf*(1.0-eta);
    shape_fcn[2 + fourj] = oneHalf*(1.0+xi) * oneHalf*(1.0+eta);
    shape_fcn[3 + fourj] = oneHalf*(1.0-xi) * oneHalf*(1.0+eta);
  }
}

void HigherOrderTri2DSCV::tri_shape_fcn_p1(
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

void HigherOrderTri2DSCV::tri_shape_fcn_p2(
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

} // namespace nalu
} // namespace Sierra