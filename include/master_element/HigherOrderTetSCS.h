/* 
 * File:   HigherOrderTet43DSCS.h
 * Author: Raphael Lindegger
 *
 * Created on November 2, 2019, 12:49 PM
 */

#ifndef HIGHERORDERTETSCS_H
#define HIGHERORDERTETSCS_H

#include <master_element/MasterElement.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/HexNElementDescription.h>
#include <element_promotion/QuadNElementDescription.h>
#include <element_promotion/TetNElementDescription.h>

#include <AlgTraits.h>
#include <KokkosInterface.h>

#include <vector>
#include <array>

namespace sierra{
namespace nalu{


struct ElementDescription;

class LagrangeBasis;
class TensorProductQuadratureRule;

class HigherOrderTetSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::gij;
  using MasterElement::face_grad_op;
  using MasterElement::adjacentNodes;

  KOKKOS_FUNCTION
  HigherOrderTetSCS(LagrangeBasis basis, TensorProductQuadratureRule quadrature);

  KOKKOS_FUNCTION
  virtual ~HigherOrderTetSCS() {}
  
  std::vector<double> getCentroid(
    std::vector<ordinal_type>& nodeOrdinals, 
    std::unique_ptr<ElementDescription>& eleDesc);

  void shape_fcn(double *shpfc) final;
  
  void hex_shape_fcn_p1( // used for the isoparametric mapping of the intgLoc on the scs
    const int npts,
    Kokkos::View<double**>& par_coord, 
    double* shape_fcn);
  
  void tet_shape_fcn_p1(
    const int npts,
    Kokkos::View<double**>& par_coord, 
    double* shape_fcn);
  
  void tet_shape_fcn_p2(
    const int npts,
    Kokkos::View<double**>& par_coord, 
    double* shape_fcn);
  
  void tet_deriv_shape_fcn_p1(
    const int npts, 
    double *deriv);
  
  void tet_deriv_shape_fcn_p2(
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

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error) final;

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv) final;

  double isInElement(
      const double *elemNodalCoord,
      const double *pointCoord,
      double *isoParCoord) final;

  void interpolatePoint(
      const int &nComp,
      const double *isoParCoord,
      const double *field,
      double *result) final;

  KOKKOS_FUNCTION const int * adjacentNodes() final;

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int ordinal = 0) const final;

  const int * side_node_ordinals(int ordinal = 0) const final;

  KOKKOS_FUNCTION int opposingNodes(
    const int ordinal, const int node) final;

  KOKKOS_FUNCTION int opposingFace(
    const int ordinal, const int node) final;

  KOKKOS_FUNCTION void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType***, DeviceShmem>& gradop) final;

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

  const double* shape_functions() const { return shapeFunctionVals_.data(); }
  const double* ip_weights() const { return ipWeights_.data(); }

private:
  void set_interior_info();
  void set_boundary_info();

  KOKKOS_FUNCTION int opposing_face_map(int k, int l, int i, int j, int face_index);

  const int nodes1D_;
  const int polyOrder_;
  const int numQuad_;
  int ipsPerFace_;
  int numSubelements_;
  const int numSubsurfacesPerSubelement_; // subsurfaces are the individual faces of the CV
  const int numSubsurfacesPerSubface_; // subsurfaces are the individual faces of the CV, faces are on the boundary of the element

  const Kokkos::View<int***> nodeMap;
  const Kokkos::View<int***> faceNodeMap;
  const Kokkos::View<int**> sideNodeOrdinals_;

  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  Kokkos::View<int**> lrscv_;
  Kokkos::View<int*> oppNode_;
  Kokkos::View<double**> shapeFunctionVals_;
  Kokkos::View<double***> shapeDerivs_;
  Kokkos::View<double***> expFaceShapeDerivs_;
  Kokkos::View<double**> intgLocSurfIso_;
  Kokkos::View<double**> intgLoc_;
  Kokkos::View<double**> intgExpFace_;
  Kokkos::View<double*> ipWeights_;
  Kokkos::View<double*> ipWeightsExpFace_;
  Kokkos::View<int*> ipNodeMap_;
  Kokkos::View<int*> oppFace_;
  std::vector<std::vector<double>> subsurfaceNodeLoc_; // internal subsurfaces, defined with 4 points
  std::vector<std::vector<double>> subsurfaceNodeLocBC_; // boundary subsurfaces, defined with 4 points
  std::vector<double> shape_fcnHex_;

  AlignedViewType<DoubleType**[3]> expRefGradWeights_;
};

} // namespace nalu
} // namespace Sierra

#endif /* HIGHERORDERTET3DSCS_H */

