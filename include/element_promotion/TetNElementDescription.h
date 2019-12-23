/* 
 * File:   TetNElementDescription.h
 * Author: Raphael Lindegger
 *
 * Created on November 2, 2019, 2:02 PM
 */

#ifndef TETNELEMENTDESCRIPTION_H
#define TETNELEMENTDESCRIPTION_H

#include <stddef.h>
#include <map>
#include <memory>
#include <vector>
#include "ElementDescription.h"

namespace sierra {
namespace nalu {

struct TetNElementDescription final: public ElementDescription
{
public:
  TetNElementDescription(std::vector<double> nodeLocs);
private:
  void set_subelement_connectivity();
  std::vector<ordinal_type> edge_node_ordinals();
  void set_edge_node_connectivities();
  std::vector<ordinal_type> face_node_ordinals();
  void set_face_node_connectivities();
  std::vector<ordinal_type> volume_node_ordinals();
  void set_volume_node_connectivities();
  void set_subelement_connectivites();
  void set_side_node_ordinals();

  std::pair<ordinal_type, ordinal_type> get_edge_offsets(
    ordinal_type i,
    ordinal_type j,
    ordinal_type k,
    ordinal_type
    edge_ordinal
  );

  std::pair<ordinal_type, ordinal_type> get_face_offsets(
    ordinal_type i,
    ordinal_type j,
    ordinal_type k,
    ordinal_type face_ordinal
  );
  void set_base_node_maps();
  void set_tensor_product_node_mappings();
  void set_boundary_node_mappings();
  void set_isoparametric_coordinates();
  std::vector<double> scaleNodeLocs(std::vector<double> in_nodeLocs);
  ordinal_type& nmap(ordinal_type i, ordinal_type j, ordinal_type k ) { return nodeMap.at( ( nodes1D*(nodes1D+1)*(nodes1D+2)/6 - (nodes1D-k)*(nodes1D-k+1)*(nodes1D-k+2)/6 ) + ( (nodes1D-k)*(nodes1D-k+1)/2 - (nodes1D-k-j)*(nodes1D-k-j+1)/2 ) + i ); };
  ordinal_type node_map(ordinal_type i, ordinal_type j, ordinal_type k) const { return nodeMap.at( ( nodes1D*(nodes1D+1)*(nodes1D+2)/6 - (nodes1D-k)*(nodes1D-k+1)*(nodes1D-k+2)/6 ) + ( (nodes1D-k)*(nodes1D-k+1)/2 - (nodes1D-k-j)*(nodes1D-k-j+1)/2 ) + i ); };
  ordinal_type node_map_bc(ordinal_type i, ordinal_type j) const { return nodeMapBC.at( (j*(nodes1D+1)-j*(j+1)/2) + i ); };
  std::vector<ordinal_type>& inmap(ordinal_type j) { return inverseNodeMap.at(j); };
  
  std::vector<double> nodeLocs1Dorig;
};

} // namespace nalu
} // namespace Sierra

#endif /* TETNELEMENTDESCRIPTION_H */

