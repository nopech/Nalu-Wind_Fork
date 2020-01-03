/* 
 * File:   HigherOrderTri3DSCS.h
 * Author: Raphael Lindegger
 *
 * Created on November 2, 2019, 12:59 PM
 */

#ifndef HIGHERORDERTRI3DSCS_H
#define HIGHERORDERTRI3DSCS_H

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

// copied from quad, not adapted to tri!
class HigherOrderTri3DSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;

  KOKKOS_FUNCTION
  HigherOrderTri3DSCS(
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);

  KOKKOS_FUNCTION
  virtual ~HigherOrderTri3DSCS() {}
  
  std::vector<double> getCentroid(
    std::vector<ordinal_type>& nodeOrdinals, 
    std::unique_ptr<ElementDescription>& eleDesc);
  
  void shape_fcn(double *shpfc) final;
  
  void quad_shape_fcn_p1( // used for the isoparametric mapping of the intgLoc on the scs
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
  
  void tri_deriv_shape_fcn_p1(
    const int npts, 
    double *deriv);
  
  void tri_deriv_shape_fcn_p2(
    const int npts,
    Kokkos::View<double**>& par_coord, 
    double *deriv);
  
  void pri_shape_fcn_p1(
    const int npts,
    Kokkos::View<double**>& par_coord, 
    double* shape_fcn);
  
  void pri_shape_fcn_p2(
    const int npts,
    Kokkos::View<double**>& par_coord, 
    double* shape_fcn);

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  const double* ip_weights() const { return ipWeights_.data(); }

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  void set_interior_info();
  void eval_shape_functions_at_ips();
  void eval_shape_derivs_at_ips();

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;
  const Kokkos::View<int**> nodeMap;
  const int nodes1D_;
  const int polyOrder_;
  const int numQuad_;
  const int numSubsurfacesPerSubelement_; // subsurfaces are the individual faces of the CV
  int numSubelements_;
  
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLocSurfIso_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
  std::vector<std::vector<double>> subsurfaceNodeLoc_; // internal subsurfaces, defined with 4 points
  std::vector<double> shape_fcnQuad_;
  int surfaceDimension_;
};

} // namespace nalu
} // namespace Sierra

#endif /* HIGHERORDERTRI3DSCS_H */

