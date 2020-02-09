/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporatlion.                                   */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <master_element/HigherOrderQuad2DSCV.h>
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
HigherOrderQuad2DSCV::HigherOrderQuad2DSCV(
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
  MasterElement::nDim_ = 2;
  nodesPerElement_ = nodes1D_*nodes1D_;

#ifndef KOKKOS_ENABLE_CUDA
  // set up integration rule and relevant maps for scvs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
#endif
}

void
HigherOrderQuad2DSCV::set_interior_info()
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
          intgLoc_(scalar_index, 0) = quadrature_.integration_point_location(k,i);
          intgLoc_(scalar_index, 1) = quadrature_.integration_point_location(l,j);
          ipWeights_(scalar_index) = quadrature_.integration_point_weight(k,l,i,j);
          ipNodeMap_(scalar_index) = nodeMap(l, k);
          
//          std::cout << std::endl;
//          std::cout << "intgLoc_ = {" << intgLoc_(scalar_index, 0) << ", " << intgLoc_(scalar_index, 1) << "}" << std::endl;
//          std::cout << "ipWeights_ = " << ipWeights_(scalar_index) << std::endl;
//          std::cout << "ipNodeMap_ = " << ipNodeMap_(scalar_index) << std::endl;

          // increment indices
          ++scalar_index;
        }
      }
    }
  }
}

void
HigherOrderQuad2DSCV::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_.data()[j];
  }
}

const int *
HigherOrderQuad2DSCV::ipNodeMap(int /*ordinal*/) const
{
  return &ipNodeMap_(0);
}

void
HigherOrderQuad2DSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  int grad_offset = 0;
  int grad_inc = nDim_ * nodesPerElement_;

  for (int ip = 0; ip < numIntPoints_; ++ip) {
    const double det_j = jacobian_determinant(coords, &shapeDerivs_.data()[grad_offset] );
    volume[ip] = ipWeights_(ip) * det_j;
    
//    std::cout << "ipWeight = " << ipWeights_(ip) << ", volume ip " << ip << " = " << volume[ip] << std::endl;

    //flag error
    if (det_j < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}

double
HigherOrderQuad2DSCV::jacobian_determinant(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDerivs) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0;
  
//  std::cout << std::endl;
//  std::cout << "new IP" << std::endl;

  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = node * nDim_;

    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];
    
//    std::cout << "xCoord = " << xCoord << ", yCoord = " << yCoord << std::endl;
    
    const double dn_ds1  = shapeDerivs[vector_offset + 0];
    const double dn_ds2  = shapeDerivs[vector_offset + 1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
  }

  const double det_j = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
  return det_j;
}

void HigherOrderQuad2DSCV::grad_op(
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

    gradient_2d(
      nodesPerElement_,
      &coords[0],
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

}  // namespace nalu
} // namespace sierra
