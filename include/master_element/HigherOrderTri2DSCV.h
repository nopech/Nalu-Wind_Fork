/* 
 * File:   HigherOrderTri2DSCV.h
 * Author: Raphael Lindegger
 *
 * Created on October 20, 2019, 10:42 AM
 */

#ifndef HigherOrderTri2DSCV_h
#define HigherOrderTri2DSCV_h

#include <master_element/MasterElement.h>

#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/NodeMapMaker.h>



#include <AlgTraits.h>
#include <KokkosInterface.h>

#include <array>

namespace sierra{
namespace nalu{

class LagrangeBasis;
class TensorProductQuadratureRule;

// This is just copied from quad, not adapted to tri!
class HigherOrderTri2DSCV final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::grad_op;

  KOKKOS_FUNCTION
  HigherOrderTri2DSCV(
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderTri2DSCV() {}

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



  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }


  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  void set_interior_info();

  double jacobian_determinant(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDerivs ) const;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;
  const Kokkos::View<int**> nodeMap;
  const int nodes1D_;

  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***>  shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
};

} // namespace nalu
} // namespace Sierra

#endif /* HIGHERORDERTRI2DSCV_H */