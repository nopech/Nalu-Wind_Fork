/* 
 * File:   TetNElementDescription.C
 * Author: Raphael Lindegger
 * 
 * Created on November 2, 2019, 2:02 PM
 */

#include <element_promotion/TetNElementDescription.h>
#include <element_promotion/TriNElementDescription.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/QuadratureRule.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/ElementDescription.h>
#include <NaluEnv.h>
#include <nalu_make_unique.h>

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <array>
#include <numeric>


namespace sierra {
namespace nalu {

TetNElementDescription::TetNElementDescription(std::vector<double> in_nodeLocs)
: ElementDescription()
{
  nodeLocs1D = scaleNodeLocs(in_nodeLocs);

  baseTopo = stk::topology::TET_4;
  polyOrder = nodeLocs1D.size()-1;
  nodes1D = nodeLocs1D.size();
  nodesPerSide = (polyOrder+2)*(polyOrder+1)/2; // Triangular number
  nodesPerElement = (polyOrder+3)*(polyOrder+2)*(polyOrder+1)/6; // Tetrahedral number
  dimension = baseTopo.dimension();
  numEdges = baseTopo.num_edges();
  numFaces = baseTopo.num_faces();
  numBoundaries = numFaces;
  nodesInBaseElement = baseTopo.num_nodes();
  nodesPerSubElement = nodesInBaseElement;

  baseEdgeConnectivity = {
      {0,1}, {1,2}, {2,0}, // bottom face
      {0,3}, {1,3}, {2,3}  // bottom-to-top
  };

  baseFaceConnectivity = {
      {1, 2, 3}, // front face
      {0, 2, 3}, // right face
      {0, 1, 3}, // left face
      {0, 1, 2}  // bottom face
  };

  baseFaceEdgeConnectivity = {
      { 0, 5, 4}, // front face
      { 2, 5, 3}, // right face
      { 0, 3, 4}, // left face
      { 0, 1, 2}  // bottom face
  };

  //first 4 nodes are base nodes.  Rest have been added.
  baseNodeOrdinals = {0,1,2,3};

  promotedNodeOrdinals.resize(nodesPerElement-nodesInBaseElement);
  std::iota(promotedNodeOrdinals.begin(), promotedNodeOrdinals.end(), nodesInBaseElement);

  newNodesPerEdge   = polyOrder - 1;
  newNodesPerFace   = (polyOrder-2)*(polyOrder-1)/2; // Triangular number with neg. signs
  newNodesPerVolume = (polyOrder-3)*(polyOrder-2)*(polyOrder-1)/6; // Tetrahedral number with neg. signs

  set_edge_node_connectivities();
  set_face_node_connectivities();
  set_volume_node_connectivities();
  set_tensor_product_node_mappings();
  set_boundary_node_mappings();
  set_side_node_ordinals();
  set_isoparametric_coordinates();
  set_subelement_connectivites();
}
//--------------------------------------------------------------------------
// Convert the isoparametric range from the quad element (-1..1) to the range of the triangle (0..1)
std::vector<double> TetNElementDescription::scaleNodeLocs(std::vector<double> in_nodeLocs)
{
  std::vector<double> nodeLocs(in_nodeLocs.size());
  
  for (std::size_t i = 0; i < in_nodeLocs.size(); ++i) {
    nodeLocs[i] = 0.5 * (in_nodeLocs[i] + 1);
  }
  
  return nodeLocs;
}
//--------------------------------------------------------------------------
std::vector<ordinal_type> TetNElementDescription::edge_node_ordinals()
{
  // base nodes -> edge nodes for node ordering
  ordinal_type numNewNodes = newNodesPerEdge * numEdges;
  std::vector<ordinal_type> edgeNodeOrdinals(numNewNodes);

  ordinal_type firstEdgeNodeNumber = nodesInBaseElement;
  std::iota(edgeNodeOrdinals.begin(), edgeNodeOrdinals.end(), firstEdgeNodeNumber);

  return edgeNodeOrdinals;
}
//--------------------------------------------------------------------------
void TetNElementDescription::set_edge_node_connectivities()
{
  std::vector<ordinal_type> edgeOrdinals(numEdges);
  std::iota(edgeOrdinals.begin(), edgeOrdinals.end(), 0);

  auto edgeNodeOrdinals = edge_node_ordinals();
  ordinal_type edgeMap[6] = {0, 1, 2, 3, 4, 5 };

  auto beginIterator = edgeNodeOrdinals.begin();
  for (const auto edgeOrdinal : edgeOrdinals) {
    auto endIterator = beginIterator + newNodesPerEdge;
    edgeNodeConnectivities.insert({edgeMap[edgeOrdinal], std::vector<ordinal_type>{beginIterator,endIterator}});
    beginIterator = endIterator;
  }
}
//--------------------------------------------------------------------------
std::vector<ordinal_type> TetNElementDescription::face_node_ordinals()
{
  // base nodes -> edge nodes for node ordering
  ordinal_type numNewFaceNodes = newNodesPerFace * numFaces;
  std::vector<ordinal_type> faceNodeOrdinals(numNewFaceNodes);

  ordinal_type firstfaceNodeNumber = nodesInBaseElement + numEdges * newNodesPerEdge;
  std::iota(faceNodeOrdinals.begin(), faceNodeOrdinals.end(), firstfaceNodeNumber);

  return faceNodeOrdinals;
}
//--------------------------------------------------------------------------
void TetNElementDescription::set_face_node_connectivities()
{
  std::vector<ordinal_type> faceOrdinals(numFaces);
  std::iota(faceOrdinals.begin(), faceOrdinals.end(), 0);

  auto faceNodeOrdinals = face_node_ordinals();

  // there's a disconnect between the exodus node ordering and face ordering,
  // the first "new" face node is entered on face #4 (bottom face, 3 in C-numbering).
  ordinal_type faceMap[4] = { 3, 1, 2, 3 };

  auto beginIterator = faceNodeOrdinals.begin();
  for (const auto faceOrdinal : faceOrdinals) {
    auto endIterator = beginIterator + newNodesPerFace;
    faceNodeConnectivities.insert({faceMap[faceOrdinal], std::vector<ordinal_type>{beginIterator,endIterator}});
    beginIterator = endIterator;
  }
}
//--------------------------------------------------------------------------
std::vector<ordinal_type> TetNElementDescription::volume_node_ordinals()
{
  // 3D volume
  ordinal_type numNewVolumeNodes = newNodesPerVolume;
  std::vector<ordinal_type> volumeNodeOrdinals(numNewVolumeNodes);

  ordinal_type firstVolumeNodeNumber = 2*(polyOrder*polyOrder+1);
  std::iota(volumeNodeOrdinals.begin(), volumeNodeOrdinals.end(), firstVolumeNodeNumber);

  return volumeNodeOrdinals;
}
//--------------------------------------------------------------------------
void TetNElementDescription::set_volume_node_connectivities()
{
  // Only 1 volume: just insert.
  volumeNodeConnectivities.insert({0, volume_node_ordinals()});
}
//--------------------------------------------------------------------------
void TetNElementDescription::set_base_node_maps()
{
  nodeMap.resize(nodesPerElement);
  inverseNodeMap.resize(nodesPerElement);

  nmap(0        , 0        , 0        ) = 0;
  nmap(polyOrder, 0        , 0        ) = 1;
  nmap(0        , polyOrder, 0        ) = 2;
  nmap(0        , 0        , polyOrder) = 3;
}
//--------------------------------------------------------------------------
void TetNElementDescription::set_boundary_node_mappings()
{
  // node mapping needs to be consistent with quad element's
  nodeMapBC = TriNElementDescription(nodeLocs1D).nodeMap;

  inverseNodeMap.resize(nodesPerElement);
  for (int i = 0; i < nodes1D; ++i) {
    for (int j = 0; j < nodes1D-i; ++j) {
      for (int k = 0; k < nodes1D-i-j; ++k) {
        inverseNodeMap[node_map(i,j,k)] = {i, j, k};
      }
    }
  }

  inverseNodeMapBC.resize(nodesPerSide);
  for (int i = 0; i < nodes1D; ++i) {
    for (int j = 0; j < nodes1D-i; ++j) {
      inverseNodeMapBC[node_map_bc(i,j)] = { i,j };
    }
  }
}
//--------------------------------------------------------------------------
void TetNElementDescription::set_tensor_product_node_mappings()
{
  set_base_node_maps();

  if (polyOrder > 1) {
    if (polyOrder == 2) {
      nmap(1, 0, 0) = 4;
      nmap(1, 1, 0) = 5;
      nmap(0, 1, 0) = 6;
      nmap(0, 0, 1) = 7;
      nmap(1, 0, 1) = 8;
      nmap(0, 1, 1) = 9;
    }
    else {
      ThrowErrorMsg("node mapping not defined for the chosen polyOrder");
    }
  }

  //inverse map
  inverseNodeMap.resize(nodesPerElement);
  for (int i = 0; i < nodes1D; ++i) {
    for (int j = 0; j < nodes1D-i; ++j) {
      for (int k = 0; k < nodes1D-i-j; ++k) {
        std::cout << "node ordinal: " << node_map(i,j,k) << std::endl;
        inverseNodeMap[node_map(i,j,k)] = {i, j, k};
      }
    }
  }
}
//--------------------------------------------------------------------------
void
TetNElementDescription::set_isoparametric_coordinates()
{
  for (int k = 0; k < nodes1D; ++k) {
    for (int j = 0; j < nodes1D-k; ++j) {
      for (int i = 0; i < nodes1D-k-j; ++i) {
        std::vector<double> nodeLoc = { nodeLocs1D.at(i), nodeLocs1D.at(j), nodeLocs1D.at(k) };
        nodeLocs.insert({node_map(i,j,k), nodeLoc});
      }
    }
  }
}
//--------------------------------------------------------------------------
void
TetNElementDescription::set_subelement_connectivites()
{
  if (polyOrder == 1) {
    subElementConnectivity.resize(1);
    subElementConnectivity[0] = 
    {
      node_map(0, 0, 0),
      node_map(1, 0, 0),
      node_map(0, 1, 0),
      node_map(0, 0, 1)
    };
  }
  else if (polyOrder == 2) {
    subElementConnectivity.resize(8);
    subElementConnectivity[0] = {0, 4, 6, 7};
    subElementConnectivity[1] = {4, 1, 5, 8};
    subElementConnectivity[2] = {5, 2, 6, 9};
    subElementConnectivity[3] = {7, 8, 9, 3};
    subElementConnectivity[4] = {7, 8, 4, 5};
    subElementConnectivity[5] = {7, 8, 9, 5};
    subElementConnectivity[6] = {7, 9, 5, 6};
    subElementConnectivity[7] = {4, 5, 6, 7};
  }
  else {
    ThrowErrorMsg("subElementConnectivity not defined for the chosen polyOrder");
  }
}
//--------------------------------------------------------------------------
void
TetNElementDescription::set_side_node_ordinals()
{
  faceNodeMap.resize(numBoundaries);
  for (int j = 0; j < numBoundaries; ++j) {
    faceNodeMap.at(j).resize(nodesPerSide);
  }
  
  if (polyOrder == 1) {
    faceNodeMap.at(0) = baseFaceConnectivity[0]; // front
    faceNodeMap.at(1) = baseFaceConnectivity[1]; // right
    faceNodeMap.at(2) = baseFaceConnectivity[2]; // left
    faceNodeMap.at(3) = baseFaceConnectivity[3]; // bottom
  }
  else if (polyOrder == 2) {
//    faceNodeMap.at(0) = {1, 5, 2, 8, 9, 3}; // front
//    faceNodeMap.at(1) = {2, 6, 0, 9, 7, 3}; // right
//    faceNodeMap.at(2) = {0, 4, 1, 7, 8, 3}; // left
//    faceNodeMap.at(3) = {0, 4, 1, 6, 5, 2}; // bottom
    
    faceNodeMap.at(0) = {1, 5, 2, 9, 3, 8}; // front
    faceNodeMap.at(1) = {0, 6, 2, 9, 3, 7}; // right
    faceNodeMap.at(2) = {0, 4, 1, 8, 3, 7}; // left
    faceNodeMap.at(3) = {0, 4, 1, 5, 2, 6}; // bottom
  }
  else {
    ThrowErrorMsg("faceNodeMap not defined for the chosen polyOrder");
  }
  
  sideOrdinalMap.resize(4);
  for (int face_ordinal = 0; face_ordinal < 4; ++face_ordinal) {
    sideOrdinalMap[face_ordinal].resize(nodesPerSide);
    for (int j = 0; j < nodesPerSide; ++j) {
      const auto& ords = inverseNodeMapBC[j];
      sideOrdinalMap.at(face_ordinal).at(j) = faceNodeMap.at(face_ordinal).at(j);
    }
  }
}

} // namespace nalu
}  // namespace sierra