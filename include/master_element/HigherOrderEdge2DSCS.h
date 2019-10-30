/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HigherOrderEdge2DSCS_h
#define HigherOrderEdge2DSCS_h

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

class HigherOrderEdge2DSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;

  KOKKOS_FUNCTION
  explicit HigherOrderEdge2DSCS(
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderEdge2DSCS() = default;

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) final;

  void shape_fcn(
    double *shpfc) final;

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }

private:
  void area_vector(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDeriv,
    std::array<double,2>& areaVector) const;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;
  const Kokkos::View<int*> nodeMap;
  const int nodes1D_;

  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***> shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
};

} // namespace nalu
} // namespace Sierra

#endif
