/* 
 * File:   HigherOrderTetSCV.C
 * Author: Raphael Lindegger
 * 
 * Created on November 2, 2019, 1:26 PM
 */

#include <master_element/HigherOrderTetSCV.h>
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
HigherOrderTetSCV::HigherOrderTetSCV(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
  : MasterElement(),
    nodes1D_(basis.order() + 1),
    numQuad_(quadrature.num_quad()*quadrature.num_quad()*quadrature.num_quad()),
    totalVol_(0.0),
    polyOrder_(nodes1D_-1)
#ifndef KOKKOS_ENABLE_CUDA
    , nodeMap(make_node_map_tet(basis.order(), true))
#endif
    , basis_(std::move(basis)),
    quadrature_(std::move(quadrature))
{
  MasterElement::nDim_ = 3;
  MasterElement::nodesPerElement_ = (polyOrder_+3)*(polyOrder_+2)*(polyOrder_+1)/6; // Tetrahedral number

#ifndef KOKKOS_ENABLE_CUDA

// generate hex shape functions used for the isoparametric mapping intgLoc on subsurfaces (scs)
  intgLocVolIso_ = Kokkos::View<double**>("integration_point_location_scv", numQuad_, 3);
  if (polyOrder_ == 1) {
    // define IP location in isoparametric subsurface
    // IP1, there is just one for P1
    intgLocVolIso_(0, 0) = quadrature_.abscissa(0);
    intgLocVolIso_(0, 1) = quadrature_.abscissa(0);
    intgLocVolIso_(0, 2) = quadrature_.abscissa(0); 
  }
  else if (polyOrder_ == 2) {
    // define IP locations in isoparametric subsurface
    // IP1
    intgLocVolIso_(0, 0) = quadrature_.abscissa(0);
    intgLocVolIso_(0, 1) = quadrature_.abscissa(0);
    intgLocVolIso_(0, 2) = quadrature_.abscissa(0);
    // IP2
    intgLocVolIso_(1, 0) = quadrature_.abscissa(1);
    intgLocVolIso_(1, 1) = quadrature_.abscissa(0);
    intgLocVolIso_(1, 2) = quadrature_.abscissa(0);
    // IP3
    intgLocVolIso_(2, 0) = quadrature_.abscissa(0);
    intgLocVolIso_(2, 1) = quadrature_.abscissa(0);
    intgLocVolIso_(2, 2) = quadrature_.abscissa(1);
    // IP4
    intgLocVolIso_(3, 0) = quadrature_.abscissa(1);
    intgLocVolIso_(3, 1) = quadrature_.abscissa(0);
    intgLocVolIso_(3, 2) = quadrature_.abscissa(1);
    // IP5
    intgLocVolIso_(4, 0) = quadrature_.abscissa(0);
    intgLocVolIso_(4, 1) = quadrature_.abscissa(1);
    intgLocVolIso_(4, 2) = quadrature_.abscissa(0);
    // IP6
    intgLocVolIso_(5, 0) = quadrature_.abscissa(1);
    intgLocVolIso_(5, 1) = quadrature_.abscissa(1);
    intgLocVolIso_(5, 2) = quadrature_.abscissa(0);
    // IP7
    intgLocVolIso_(6, 0) = quadrature_.abscissa(0);
    intgLocVolIso_(6, 1) = quadrature_.abscissa(1);
    intgLocVolIso_(6, 2) = quadrature_.abscissa(1);
    // IP8
    intgLocVolIso_(7, 0) = quadrature_.abscissa(1);
    intgLocVolIso_(7, 1) = quadrature_.abscissa(1);
    intgLocVolIso_(7, 2) = quadrature_.abscissa(1);
  }
  else {
    ThrowErrorMsg("Only P1 and P2 is defined for TET_4 elements.");
  }
  
  shape_fcnHex_.resize(numQuad_ * 8);
  double *p_shape_fcnHex = &shape_fcnHex_[0];
  hex_shape_fcn_p1(numQuad_, intgLocVolIso_, &p_shape_fcnHex[0]);
  
  set_interior_info();
#endif
}

std::vector<double> 
HigherOrderTetSCV::getCentroid(std::vector<ordinal_type>& nodeOrdinals, std::unique_ptr<ElementDescription>& eleDesc) {
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
HigherOrderTetSCV::set_interior_info()
{
  auto desc = ElementDescription::create(3, polyOrder_, stk::topology::TET_4);
  
  numSubelements_ = desc->subElementConnectivity.size();
  numIntPoints_ = numSubelements_ * 4 * numQuad_;
  ipNodeMap_= Kokkos::View<int*>("ipNodeMap_", numIntPoints_);
  intgLoc_ = Kokkos::View<double**>("integration_point_location", numIntPoints_, 3);
  ipWeights_ = Kokkos::View<double*>("ip_weight", numIntPoints_);
  subvolNodeLoc_.resize(numSubelements_ * 4 * 8, std::vector<double>(3));
  ordinal_type subcontrol_vol_ord;

  int countIP = 0;
  std::vector<std::vector<int>> subvolCreationIndices {
    {0, 1, 3}, // scv 0
    {0, 1},
    {0},
    {0, 3},
    {0, 1, 2, 3},
    {0, 1, 2},
    {0, 2},
    {0, 2, 3}, // end
    {1, 2, 3}, // scv 1
    {1, 2},
    {1},
    {1, 3},
    {0, 1, 2, 3},
    {0, 1, 2},
    {0, 1},
    {0, 1, 3}, // end
    {0, 2, 3}, // scv 2
    {0, 2},
    {2},
    {2, 3},
    {0, 1, 2, 3},
    {0, 1, 2},
    {1, 2},
    {1, 2, 3}, // end
    {0, 1, 3}, // scv 3
    {0, 3},
    {3},
    {1, 3},
    {0, 1, 2, 3},
    {0, 2, 3},
    {2, 3},
    {1, 2, 3} // end 
  };

  // initialize intgLoc_
  for (int i = 0; i < numIntPoints_; ++i) {
    for (int j = 0; j < nDim_; ++j) {
      intgLoc_(i, j) = 0.0;
    }
  }
   
  // loop through each subelement and compute the integration points at each scv in the subelement
  int countNode = 0;
  for (int subElement = 0; subElement < numSubelements_; ++subElement) {
    
    int countSubvol = 0;
    for (int subVol = 0; subVol < 4; ++subVol) {
//      std::cout << std::endl;
//      std::cout << "new scs" << std::endl;
      
      for (int node = 0; node < 8; ++node) {
        const int numOrd = subvolCreationIndices[countSubvol].size();
        std::vector<ordinal_type> centroidDefiningOrdinals(numOrd);
   
        for (int i = 0; i < numOrd; ++i) {
          const int ordIndex = subvolCreationIndices[countSubvol][i];
          centroidDefiningOrdinals[i] = desc->subElementConnectivity[subElement][ordIndex];
        }
        
        if (numOrd == 1) {
          subcontrol_vol_ord = centroidDefiningOrdinals[0];
        }
        
        // compute scv node location and save it for later usage in volume computation
        std::vector<double> nodeLoc = getCentroid(centroidDefiningOrdinals, desc);
        
        int subvolNodeLocIndex = 32*subElement + 8*subVol + node;
        subvolNodeLoc_[subvolNodeLocIndex] = nodeLoc;
//        std::cout << "nodeLoc = {" << nodeLoc[0] << ", " << nodeLoc[1] << ", " << nodeLoc[2] << "}" << std::endl;
        
        countSubvol++;
      }

      // isoparametric mapping of the intgLoc of a isoparametric rectangle to the isoparametric tet
      int countHexSF = 0;;
      int quadIndex = 0;
      for (int quadPoint = 0; quadPoint < numQuad_; ++quadPoint) { // for each ip in scv
//        std::cout << "new quadpoint" << std::endl;
        
        if (quadIndex >= quadrature_.num_quad()) {
          quadIndex = 0;
        }
        
        // IP weight
        ipWeights_(countIP) = quadrature_.weights(quadIndex) * quadrature_.weights(quadIndex) * quadrature_.weights(quadIndex);
        ipNodeMap_(countIP) = subcontrol_vol_ord;
          
          for (int i = 0; i < 8; ++i) { // for each node of the subsurf
            int subvolNodeLocIndex = 32*subElement + 8*subVol + i;
            
            for (int j = 0; j < 3; ++j) { // for each dimension
              intgLoc_(countIP, j) += (shape_fcnHex_[countHexSF] * subvolNodeLoc_[subvolNodeLocIndex][j]);
            }
            
            countHexSF++;
          }
        
//        std::cout << "isoCalc intgLoc: " << intgLoc_(countIP, 0) << ", " << intgLoc_(countIP, 1) << ", " << intgLoc_(countIP, 2) << std::endl;
        countIP++;
        quadIndex++;
        
      } // ip
    } // subVol
  } // subElement 
}

void
HigherOrderTetSCV::shape_fcn(double *shpfc)
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

const int* HigherOrderTetSCV::ipNodeMap(int) const { return ipNodeMap_.data(); }

void HigherOrderTetSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  const int numSubVol = numIntPoints_/numQuad_;
  const int ipsPerSubvol = numQuad_;
  Kokkos::View<double**> realCoords;
  Kokkos::View<double**> isoParCoords;
  realCoords = Kokkos::View<double[8][3]>("realCoords");
  isoParCoords = Kokkos::View<double[8][3]>("isoParCoords");
  std::vector<double> shape_fcn(8 * nodesPerElement_);
  double *p_shape_fcn = &shape_fcn[0];
  
  auto desc = ElementDescription::create(3, polyOrder_, stk::topology::TET_4);

  // loop through all scv
  int countIP = 0;
  for (int subVol = 0; subVol < numSubVol; ++subVol) {
    
    // initialize coords vectors
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 3; ++j) {
        realCoords(i, j) = 0.0;
        isoParCoords(i, j) = subvolNodeLoc_[subVol*8 + i][j];
      }
    }
    
    // evaluate shape functions at the vertices of the scs
    if (polyOrder_ == 1) {
      tet_shape_fcn_p1(8, isoParCoords, &p_shape_fcn[0]);
    }
    else if (polyOrder_ == 2) {
      tet_shape_fcn_p2(8, isoParCoords, &p_shape_fcn[0]);
    }
    else {
    ThrowErrorMsg("Shape functions not defined for the chosen polyOrder");
    }
    
    // use isoparametric mapping to get real coordinates of the vertices
    int count = 0;
    for (int vert = 0; vert < 8; ++vert) {
      for (int node = 0; node < nodesPerElement_; ++node) {
        for (int j = 0; j < 3; ++j) {
          realCoords(vert, j) += (shape_fcn[count] * coords[node * nDim_ + j]);
        }
        count++;
      }
    }
    
    const double vol_scv = std::abs(hex_volume_grandy(realCoords));
//    totalVol_ += vol_scv;
    
    // Loop through all quad points of the current scv
    for (int quadPoint = 0; quadPoint < numQuad_; ++quadPoint) {
      volume[countIP] = ipWeights_(countIP) * vol_scv;
//      std::cout << "ipWeight = " << ipWeights_(countIP) << ", " << "vol_scv = " << vol_scv << std::endl;
      countIP++;
    }
  }
  *error = 0; // no error checking available
//  std::cout << "total volume = " << totalVol_ << std::endl;
}

void HigherOrderTetSCV::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  *error = 0.0;
}

void
HigherOrderTetSCV::hex_shape_fcn_p1(
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

void HigherOrderTetSCV::tet_shape_fcn_p1(
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

void HigherOrderTetSCV::tet_shape_fcn_p2(
  const int   npts,
  Kokkos::View<double**>& par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int tenj = 10*j;
    const double xi = par_coord(j, 0);
    const double eta = par_coord(j, 1);
    const double zeta = par_coord(j, 2);
    
    const double L1 = 1.0-xi-eta-zeta;
    const double L2 = xi;
    const double L3 = eta;
    const double L4 = zeta;
    
    shape_fcn[0 + tenj] = L1*(2.0*L1-1.0);
    shape_fcn[1 + tenj] = L2*(2.0*L2-1.0);
    shape_fcn[2 + tenj] = L3*(2.0*L3-1.0);
    shape_fcn[3 + tenj] = L4*(2.0*L4-1.0);
    shape_fcn[4 + tenj] = 4.0*L1*L2;
    shape_fcn[5 + tenj] = 4.0*L2*L3;
    shape_fcn[6 + tenj] = 4.0*L3*L1;
    shape_fcn[7 + tenj] = 4.0*L1*L4;
    shape_fcn[8 + tenj] = 4.0*L2*L4;
    shape_fcn[9 + tenj] = 4.0*L3*L4;
  }
}

double HigherOrderTetSCV::hex_volume_grandy(Kokkos::View<double**> scvCoords)
{
  /**
   * The Grandy algorithm for computing the volume of a multilinear box
   *
   * "Efficient computation of volume ofl
   * Hexahedral Cells", Jeffrey Grandy, LLNL, UCRL-ID-128886,
   *  October 30, 1997.
   */
  constexpr int nTri = 24;
  constexpr int dim = 3;

  constexpr int nNodes = 8;
  constexpr int nFaces = 6;
  constexpr int npv = nNodes + nFaces;

  double coordv[npv][dim];

  // copy coordinates
  for (int n = 0; n < nNodes; ++n) {
    coordv[n][0] = scvCoords(n, 0);
    coordv[n][1] = scvCoords(n, 1);
    coordv[n][2] = scvCoords(n, 2);
  }

  constexpr int nodesPerFace = 4;
  constexpr int face_nodes[nFaces][nodesPerFace] = {
      { 0, 3, 2, 1 }, { 4, 5, 6, 7 },
      { 0, 1, 5, 4 }, { 2, 3, 7, 6 },
      { 1, 2, 6, 5 }, { 0, 4, 3, 7 }
  };

  // append face midpoint coordinates
  for (int k = 0; k < nFaces; ++k) {
    const int coordIndex = k + nNodes;
    for (int d = 0; d < dim; ++d) {
      coordv[coordIndex][d] = 0.25 * (
           coordv[face_nodes[k][0]][d]
         + coordv[face_nodes[k][1]][d]
         + coordv[face_nodes[k][2]][d]
         + coordv[face_nodes[k][3]][d]
      );
    }
  }

  constexpr int triangular_facets[nTri][3] = {
      { 0,  8,  1}, { 8,  2,  1}, { 3,  2,  8},
      { 3,  8,  0}, { 6,  9,  5}, { 7,  9,  6},
      { 4,  9,  7}, { 4,  5,  9}, {10,  0,  1},
      { 5, 10,  1}, { 4, 10,  5}, { 4,  0, 10},
      { 7,  6, 11}, { 6,  2, 11}, { 2,  3, 11},
      { 3,  7, 11}, { 6, 12,  2}, { 5, 12,  6},
      { 5,  1, 12}, { 1,  2, 12}, { 0,  4, 13},
      { 4,  7, 13}, { 7,  3, 13}, { 3,  0, 13}
  };

  double volume = 0.0;
  for (int k = 0; k < nTri; ++k) {
    const int p = triangular_facets[k][0];
    const int q = triangular_facets[k][1];
    const int r = triangular_facets[k][2];

    const double triFaceMid[3] = {
        coordv[p][0] + coordv[q][0] + coordv[r][0],
        coordv[p][1] + coordv[q][1] + coordv[r][1],
        coordv[p][2] + coordv[q][2] + coordv[r][2]
    };

    enum {XC = 0, YC = 1, ZC = 2};
    double dxv[3];

    dxv[0] = ( coordv[q][YC] - coordv[p][YC] ) * ( coordv[r][ZC] - coordv[p][ZC] )
           - ( coordv[r][YC] - coordv[p][YC] ) * ( coordv[q][ZC] - coordv[p][ZC] );

    dxv[1] = ( coordv[r][XC] - coordv[p][XC] ) * ( coordv[q][ZC] - coordv[p][ZC] )
           - ( coordv[q][XC] - coordv[p][XC] ) * ( coordv[r][ZC] - coordv[p][ZC] );

    dxv[2] = ( coordv[q][XC] - coordv[p][XC] ) * ( coordv[r][YC] - coordv[p][YC] )
           - ( coordv[r][XC] - coordv[p][XC] ) * ( coordv[q][YC] - coordv[p][YC] );

    volume += triFaceMid[0] * dxv[0] + triFaceMid[1] * dxv[1] + triFaceMid[2] * dxv[2];
  }
  volume /= double(18.0);
  return volume;
}

}  // namespace nalu
} // namespace sierra
