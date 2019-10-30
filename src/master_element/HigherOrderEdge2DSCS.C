/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporatlion.                                   */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <master_element/HigherOrderEdge2DSCS.h>
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
HigherOrderEdge2DSCS::HigherOrderEdge2DSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(basis),
  quadrature_(quadrature),
#ifndef KOKKOS_ENABLE_CUDA
  nodeMap(make_node_map_edge(basis.order())),
#endif
  nodes1D_(basis.order()+1)
{
  MasterElement::nDim_ = 2;
  nodesPerElement_ = nodes1D_;
  numIntPoints_ = quadrature_.num_quad() * nodes1D_;

#ifndef KOKKOS_ENABLE_CUDA
  ipNodeMap_= Kokkos::View<int*>("ipNodeMap_", numIntPoints_);
  intgLoc_ =  Kokkos::View<double*[1]>("intgLoc_", numIntPoints_);
  ipWeights_ = Kokkos::View<double*>("ipWeights_",numIntPoints_);

  int scalar_index = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    for (int i = 0; i < quadrature_.num_quad(); ++i) {
      // TODO(psakiev) double check this
      intgLoc_(scalar_index, 0)  = quadrature_.integration_point_location(k,i);
      std::cout << "intgLoc_: " << quadrature_.integration_point_location(k,i) << std::endl;
      ipWeights_(scalar_index) = quadrature_.integration_point_weight(k,i);
      std::cout << "ipWeights_: " << quadrature_.integration_point_weight(k,i) << std::endl;
      ipNodeMap_(scalar_index) = nodeMap(k);
      ++scalar_index;
    }
  }
  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
#endif
}

const int *
HigherOrderEdge2DSCS::ipNodeMap(int /*ordinal*/) const
{
  return ipNodeMap_.data();
}

void
HigherOrderEdge2DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  std::array<double,2> areaVector;
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  int grad_offset = 0;
  const int grad_inc = nodesPerElement_;

  int vec_offset = 0;
  for (int ip = 0; ip < numIntPoints_; ++ip) {
    // calculate the area vector
    area_vector( &coords[0],
      &shapeDerivs_.data()[grad_offset],
      areaVector );

    // weight the area vector with the Gauss-quadrature weight for this IP
    areav[vec_offset + 0] = ipWeights_(ip) * areaVector[0];
    areav[vec_offset + 1] = ipWeights_(ip) * areaVector[1];

    grad_offset += grad_inc;
    vec_offset += nDim_;
  }
}

void
HigherOrderEdge2DSCS::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
   for (int j = 0; j < numShape; ++j) {
     shpfc[j] = shapeFunctionVals_.data()[j];
   }
}

void
HigherOrderEdge2DSCS::area_vector(
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT shapeDeriv,
  std::array<double,2>& areaVector) const
{
  double dxdr = 0.0;  double dydr = 0.0;
  int vector_offset = 0;
  for (int node = 0; node < nodesPerElement_; ++node) {
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];

    dxdr += shapeDeriv[node] * xCoord;
    dydr += shapeDeriv[node] * yCoord;

    vector_offset += nDim_;
  }
  areaVector[0] =  dydr;
  areaVector[1] = -dxdr;
}

}  // namespace nalu
} // namespace sierra
