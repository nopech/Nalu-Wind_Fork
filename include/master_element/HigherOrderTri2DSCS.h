/* 
 * File:   HigherOrderTri2DSCS.h
 * Author: Raphael Lindegger
 *
 * Created on October 20, 2019, 10:43 AM
 */

#ifndef HigherOrderTri2DSCS_h
#define HigherOrderTri2DSCS_h

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

class HigherOrderTri2DSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::grad_op;

  KOKKOS_FUNCTION
  HigherOrderTri2DSCS(
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderTri2DSCS() {}
  
  std::vector<double> getCentroid(
    std::vector<ordinal_type> nodeOrdinals, 
    std::unique_ptr<ElementDescription>& eleDesc);

  void shape_fcn(double *shpfc) final;
  
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

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error) final;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error) final;

  KOKKOS_FUNCTION const int * adjacentNodes() final;

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  const int * side_node_ordinals(int ordinal = 0) const final;

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  Kokkos::View<int*> lrscv_;

  void set_interior_info();
  void set_boundary_info();

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;
  const Kokkos::View<int**> nodeMap;
  const Kokkos::View<int**> faceNodeMap;
  const Kokkos::View<int**> sideNodeOrdinals_;
  const int nodes1D_;
  const int polyOrder_;
  const int numQuad_;

  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***>  shapeDerivs_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<int*> ipNodeMap_;
  int ipsPerFace_;
  Kokkos::View<double***> expFaceShapeDerivs_;
  Kokkos::View<int*> oppNode_;
  Kokkos::View<int*> oppFace_;
  Kokkos::View<double**> intgExpFace_;
  Kokkos::View<double***>  intSubfaces_; // internal subfaces, defined with 2 points B and P
};

} // namespace nalu
} // namespace Sierra

#endif /* HIGHERORDERTRI2DSCS_H */