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
#include <element_promotion/ElementDescription.h>



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
  
  std::vector<double> getCentroid(
    std::vector<ordinal_type> nodeOrdinals, 
    std::unique_ptr<ElementDescription>& eleDesc);
  
  void quad_shape_fcn_p1( // used for the isoparametric mapping of the intgLoc on the scv
    const int npts,
    Kokkos::View<double**>& par_coord, 
    double* shape_fcn);
  
  void tri_shape_fcn_p1(
    const int npts,
    Kokkos::View<double**>& par_coord, 
    double* shape_fcn);
  
  void tri_shape_fcn_p2(
    const int npts,
    Kokkos::View<double**>& par_coord, 
    double* shape_fcn);

  void shape_fcn(double *shpfc) final;

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *volume,
    double * error ) final;

  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }


  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  void set_interior_info();

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;
  const Kokkos::View<int**> nodeMap;
  const int nodes1D_;
  const int polyOrder_;
  const int numQuad_;
  double totalVol_;

  Kokkos::View<double**> intgLocSurfIso_;
  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***>  shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
  Kokkos::View<double***> intSubVolumes_; // internal subvolumes, defined with 4 points
  
  std::vector<double> shape_fcnQuad_;
};

} // namespace nalu
} // namespace Sierra

#endif /* HIGHERORDERTRI2DSCV_H */