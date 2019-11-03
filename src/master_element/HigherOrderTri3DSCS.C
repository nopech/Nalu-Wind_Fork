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
#include <element_promotion/ElementDescription.h>

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
  nodeMap(make_node_map_quad(basis.order())),
#endif
  nodes1D_(basis.order()+1)
{
  surfaceDimension_ = 2;
  MasterElement::nDim_ = 3;
  nodesPerElement_ = nodes1D_*nodes1D_;

#ifndef KOKKOS_ENABLE_CUDA
  // set up integration rule and relevant maps on scs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
#endif
}

void
HigherOrderTri3DSCS::set_interior_info()
{
  //1D integration rule per sub-control volume
  numIntPoints_ = (nodesPerElement_) * ( quadrature_.num_quad() * quadrature_.num_quad() );

  // define ip node mappings
  ipNodeMap_= Kokkos::View<int*>("ipNodeMap_", numIntPoints_);
  intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
  ipWeights_ = Kokkos::View<double*>("ipWeights_",numIntPoints_);

  // tensor product nodes x tensor product quadrature
  int scalar_index = 0;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      for (int j = 0; j < quadrature_.num_quad(); ++j) {
        for (int i = 0; i < quadrature_.num_quad(); ++i) {
          //integration point location
          intgLoc_(scalar_index, 0) = quadrature_.integration_point_location(k,i);
          intgLoc_(scalar_index, 1) = quadrature_.integration_point_location(l,j);

          //weight
          ipWeights_(scalar_index) = quadrature_.integration_point_weight(k,l,i,j);

          //sub-control volume association
          ipNodeMap_(scalar_index) = nodeMap(l,k);

          // increment indices
          ++scalar_index;
        }
      }
    }
  }
}

void
HigherOrderTri3DSCS::shape_fcn(double* shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_.data()[j];
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

  std::array<double,3> areaVector;
  int grad_offset = 0;
  int grad_inc = surfaceDimension_ * nodesPerElement_;

  int vector_offset = 0;
  for (int ip = 0; ip < numIntPoints_; ++ip) {
    //compute area vector for this ip
    area_vector( &coords[0], &shapeDerivs_.data()[grad_offset], areaVector );

    // apply quadrature weight and orientation (combined as weight)
    for (int j = 0; j < nDim_; ++j) {
      areav[vector_offset+j]  = ipWeights_(ip) * areaVector[j];
    }
    vector_offset += nDim_;
    grad_offset += grad_inc;
  }
}

void
HigherOrderTri3DSCS::area_vector(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDeriv,
  std::array<double,3>& areaVector) const
{
  // return the normal area vector given shape derivatives dnds OR dndt
  double dx_ds1 = 0.0; double dy_ds1 = 0.0; double dz_ds1 = 0.0;
  double dx_ds2 = 0.0; double dy_ds2 = 0.0; double dz_ds2 = 0.0;

  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const int surface_vector_offset = surfaceDimension_ * node;

    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];
    const double zCoord = elemNodalCoords[vector_offset+2];

    const double dn_ds1 = shapeDeriv[surface_vector_offset+0];
    const double dn_ds2 = shapeDeriv[surface_vector_offset+1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
  }

  //cross product
  areaVector[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
  areaVector[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
  areaVector[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}

}  // namespace nalu
} // namespace sierra