/* 
 * File:   HigherOrderTetSCV.h
 * Author: Raphael Lindegger
 *
 * Created on November 2, 2019, 1:26 PM
 */

#ifndef HIGHERORDERTETSCV_H
#define HIGHERORDERTETSCV_H

#include <master_element/MasterElement.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/HexNElementDescription.h>
#include <element_promotion/QuadNElementDescription.h>

#include <AlgTraits.h>
#include <KokkosInterface.h>

#include <vector>
#include <array>

namespace sierra{
namespace nalu{


struct ElementDescription;

class LagrangeBasis;
class TensorProductQuadratureRule;

class HigherOrderTetSCV final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;

  KOKKOS_FUNCTION
  HigherOrderTetSCV(LagrangeBasis basis, TensorProductQuadratureRule quadrature);

  KOKKOS_FUNCTION
  virtual ~HigherOrderTetSCV() {}

  void shape_fcn(double *shpfc) final;
  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *volume,
    double * error ) final;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error) final;

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }

private:
  void set_interior_info();

  double jacobian_determinant(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDerivs ) const;

  const int nodes1D_;
  const Kokkos::View<int***> nodeMap;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***> shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
};

} // namespace nalu
} // namespace Sierra

#endif /* HIGHERORDERTETSCV_H */

