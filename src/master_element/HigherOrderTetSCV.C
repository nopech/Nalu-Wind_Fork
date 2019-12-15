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
    nodes1D_(basis.order() + 1)
#ifndef KOKKOS_ENABLE_CUDA
    , nodeMap(make_node_map_tet(basis.order(), true))
#endif
    , basis_(std::move(basis)),
    quadrature_(std::move(quadrature))
{
  MasterElement::nDim_ = 3;
  int polyOrder = nodes1D_-1;
  MasterElement::nodesPerElement_ = (polyOrder+3)*(polyOrder+2)*(polyOrder+1)/6; // Tetrahedral number
  MasterElement::numIntPoints_ = nodesPerElement_ * (quadrature_.num_quad() * quadrature_.num_quad() * quadrature_.num_quad());

#ifndef KOKKOS_ENABLE_CUDA
  ipNodeMap_ = Kokkos::View<int*>("ip_node_map", numIntPoints_);
  intgLoc_ = Kokkos::View<double**>("integration_point_location", numIntPoints_, 3);
  ipWeights_ = Kokkos::View<double*>("integration_point_weights", numIntPoints_);

//  int flat_index = 0;
//  for (int n = 0; n < nodes1D_; ++n) {
//    for (int m = 0; m < nodes1D_; ++m) {
//      for (int l = 0; l < nodes1D_; ++l) {
//        for (int k = 0; k < quadrature_.num_quad(); ++k) {
//          for (int j = 0; j < quadrature_.num_quad(); ++j) {
//            for (int i = 0; i < quadrature_.num_quad(); ++i) {
//              intgLoc_(flat_index, 0)= quadrature_.integration_point_location(l,i);
//              intgLoc_(flat_index, 1) = quadrature_.integration_point_location(m,j);
//              intgLoc_(flat_index, 2) = quadrature_.integration_point_location(n,k);
//              ipWeights_[flat_index] = quadrature_.integration_point_weight(l, m, n, i, j, k);
//              ipNodeMap_[flat_index] = nodeMap(n, m, l);
//              ++flat_index;
//            }
//          }
//        }
//      }
//    }
//  }
//  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
//  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
#endif
}

void
HigherOrderTetSCV::shape_fcn(double *shpfc)
{
//  int numShape = shapeFunctionVals_.size();
//  for (int j = 0; j < numShape; ++j) {
//    shpfc[j] = shapeFunctionVals_.data()[j];
//  }
}

const int* HigherOrderTetSCV::ipNodeMap(int) const { return ipNodeMap_.data(); }

void HigherOrderTetSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  *error = 0.0;
//  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");
//
//  int grad_offset = 0;
//  const int grad_inc = nDim_ * nodesPerElement_;
//
//  for (int ip = 0; ip < numIntPoints_; ++ip, grad_offset += grad_inc) {
//    const double det_j = jacobian_determinant(coords,  &shapeDerivs_.data()[grad_offset]);
//    volume[ip] = ipWeights_[ip] * det_j;
//
//    if (det_j < tiny_positive_value()) {
//      *error = 1.0;
//    }
//  }
}

double HigherOrderTetSCV::jacobian_determinant(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDerivs) const
{
//  double dx_ds1 = 0.0;  double dx_ds2 = 0.0; double dx_ds3 = 0.0;
//  double dy_ds1 = 0.0;  double dy_ds2 = 0.0; double dy_ds3 = 0.0;
//  double dz_ds1 = 0.0;  double dz_ds2 = 0.0; double dz_ds3 = 0.0;
//  for (int node = 0; node < nodesPerElement_; ++node) {
//    const int vector_offset = nDim_ * node;
//
//    const double xCoord = elemNodalCoords[vector_offset+0];
//    const double yCoord = elemNodalCoords[vector_offset+1];
//    const double zCoord = elemNodalCoords[vector_offset+2];
//
//    const double dn_ds1 = shapeDerivs[vector_offset+0];
//    const double dn_ds2 = shapeDerivs[vector_offset+1];
//    const double dn_ds3 = shapeDerivs[vector_offset+2];
//
//    dx_ds1 += dn_ds1 * xCoord;
//    dx_ds2 += dn_ds2 * xCoord;
//    dx_ds3 += dn_ds3 * xCoord;
//
//    dy_ds1 += dn_ds1 * yCoord;
//    dy_ds2 += dn_ds2 * yCoord;
//    dy_ds3 += dn_ds3 * yCoord;
//
//    dz_ds1 += dn_ds1 * zCoord;
//    dz_ds2 += dn_ds2 * zCoord;
//    dz_ds3 += dn_ds3 * zCoord;
//  }
//
//  const double det_j = dx_ds1 * ( dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3 )
//                     + dy_ds1 * ( dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3 )
//                     + dz_ds1 * ( dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3 );

  return 0; //det_j;
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
//  ThrowRequireMsg(nelem == 1, "Grad_op is executed one element at a time for HO");
//
//  int grad_offset = 0;
//  int grad_inc = nDim_ * nodesPerElement_;
//
//  for (int ip = 0; ip < numIntPoints_; ++ip) {
//    for (int j = 0; j < grad_inc; ++j) {
//      deriv[grad_offset + j] = shapeDerivs_.data()[grad_offset +j];
//    }
//
//    gradient_3d(nodesPerElement_, coords, &shapeDerivs_.data()[grad_offset], &gradop[grad_offset], &det_j[ip]);
//
//    if (det_j[ip] < tiny_positive_value()) {
//      *error = 1.0;
//    }
//
//    grad_offset += grad_inc;
//  }
}

}  // namespace nalu
} // namespace sierra
