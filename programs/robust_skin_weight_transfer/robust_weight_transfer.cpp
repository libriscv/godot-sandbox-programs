#include <Eigen/Dense>
#include <cstdint>
#include <vector>
#include <iostream>
#include <set>

#include <api.hpp>
#include <Eigen/Dense>

#include <vector>
#include <tuple>

#include <igl/point_mesh_squared_distance.h>
#include <igl/barycentric_coordinates.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/adjacency_list.h>
#include <igl/invert_diag.h>
#include <igl/slice_mask.h>
#include <igl/min_quad_with_fixed.h>

#include "generated_api.hpp"

/**
 * Given a number of points find their closest points on the surface of the V,F mesh
 * 
 *  P: #P by 3, where every row is a point coordinate
 *  V: #V by 3 mesh vertices
 *  F: #F by 3 mesh triangles indices
 *  sqrD #P smallest squared distances
 *  I #P primitive indices corresponding to smallest distances
 *  C #P by 3 closest points
 *  B #P by 3 of the barycentric coordinates of the closest point
 */
void find_closest_point_on_surface(const Eigen::MatrixXd& P, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, 
                                   Eigen::VectorXd& sqrD, Eigen::VectorXi& I, Eigen::MatrixXd& C, Eigen::MatrixXd& B)
{
    igl::point_mesh_squared_distance(P, V, F, sqrD, I, C);

    Eigen::MatrixXi F_closest = F(I, Eigen::indexing::all);
    Eigen::MatrixXd V1 = V(F_closest(Eigen::indexing::all, 0), Eigen::indexing::all);
    Eigen::MatrixXd V2 = V(F_closest(Eigen::indexing::all, 1), Eigen::indexing::all);
    Eigen::MatrixXd V3 = V(F_closest(Eigen::indexing::all, 2), Eigen::indexing::all);

    igl::barycentric_coordinates(C, V1, V2, V3, B);
}

/** 
 * Interpolate per-vertex attributes A via barycentric coordinates B of the F[I,:] vertices
 * 
 *  A: #V by N per-vertex attributes
 *  B  #B by 3 array of the barycentric coordinates of some points
 *  I  #B primitive indices containing the closest point
 *  F: #F by 3 mesh triangle indices
 *  A_out #B interpolated attributes
 */
void interpolate_attribute_from_bary(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                     const Eigen::VectorXi& I, const Eigen::MatrixXi& F, 
                                     Eigen::MatrixXd& A_out)
{
    Eigen::MatrixXi F_closest = F(I, Eigen::indexing::all);

    Eigen::MatrixXd a1 = A(F_closest.col(0), Eigen::indexing::all);
    Eigen::MatrixXd a2 = A(F_closest.col(1), Eigen::indexing::all);
    Eigen::MatrixXd a3 = A(F_closest.col(2), Eigen::indexing::all);

    Eigen::VectorXd b1 = B.col(0);
    Eigen::VectorXd b2 = B.col(1);
    Eigen::VectorXd b3 = B.col(2);

    a1 = a1.array().colwise() * b1.array();
    a2 = a2.array().colwise() * b2.array();
    a3 = a3.array().colwise() * b3.array();

    A_out = a1 + a2 + a3;
}

Eigen::VectorXd normalize_vector(const Eigen::VectorXd& p_vector) {
    return p_vector.normalized();
}


/**
 * For each vertex on the target mesh find a match on the source mesh.
 * 
 *  V1: #V1 by 3 source mesh vertices
 *  F1: #F1 by 3 source mesh triangles indices
 *  N1: #V1 by 3 source mesh normals
 *  V2: #V2 by 3 target mesh vertices
 *  F2: #F2 by 3 target mesh triangles indices
 *  N2: #V2 by 3 target mesh normals
 *  W1: #V1 by num_bones source mesh skin weights
 *  dDISTANCE_THRESHOLD_SQRD: distance threshold
 *  dANGLE_THRESHOLD_DEGREES: normal threshold
 *  Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
 *  W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
 */
void find_matches_closest_surface(const Eigen::MatrixXd& V1, const Eigen::MatrixXi& F1, const Eigen::MatrixXd& N1, 
                                  const Eigen::MatrixXd& V2, const Eigen::MatrixXi& F2, const Eigen::MatrixXd& N2, 
                                  const Eigen::MatrixXd& W1, 
                                  double dDISTANCE_THRESHOLD_SQRD, 
                                  double dANGLE_THRESHOLD_DEGREES,
                                  Eigen::MatrixXd& W2,
                                  Eigen::Array<bool,Eigen::Dynamic,1>& Matched)
{
    Matched = Eigen::Array<bool,Eigen::Dynamic,1>::Constant(V2.rows(), false);
    Eigen::VectorXd sqrD; 
    Eigen::VectorXi I;
    Eigen::MatrixXd C, B;
    find_closest_point_on_surface(V2, V1, F1, sqrD, I, C, B);
    
    // for each closest point on the source, interpolate its per-vertex attributes(skin weights and normals) 
    // using the barycentric coordinates
    interpolate_attribute_from_bary(W1, B, I, F1, W2);

    Eigen::MatrixXd N1_match_interpolated;
    interpolate_attribute_from_bary(N1, B, I, F1, N1_match_interpolated);
    
    // check that the closest point passes our distance and normal thresholds
    for (int RowIdx = 0; RowIdx < V2.rows(); ++RowIdx)
    {
        Eigen::VectorXd n1 = N1_match_interpolated.row(RowIdx);
        n1.normalize();

        Eigen::VectorXd n2 = N2.row(RowIdx);
        n2.normalize();

        const double rad_angle = acos(n1.dot(n2));
        const double deg_angle = rad_angle * (180.0 / M_PI);

        if (sqrD(RowIdx) <= dDISTANCE_THRESHOLD_SQRD and deg_angle <= dANGLE_THRESHOLD_DEGREES)
        {
            Matched(RowIdx) = true;
        }
    }
}

bool is_valid_array(const Eigen::MatrixXd& p_matrix) {
    return p_matrix.allFinite();
}

/**
 * Inpaint weights for all the vertices on the target mesh for which  we didnt 
 * find a good match on the source (i.e. Matched[i] == False).
 * 
 *  V2: #V2 by 3 target mesh vertices
 *  F2: #F2 by 3 target mesh triangles indices
 *  W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
 *  Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
 *  W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
 *  success: true if inpainting succeeded, false otherwise
 */
bool inpaint(const Eigen::MatrixXd& p_V2, const Eigen::MatrixXi& p_F2, const Eigen::MatrixXd& p_W2, const Eigen::Array<bool,Eigen::Dynamic,1>& p_Matched, Eigen::MatrixXd& r_W_inpainted)
{
    // Compute the laplacian
    Eigen::SparseMatrix<double> L, M, Minv;
    igl::cotmatrix(p_V2, p_F2, L);
    igl::massmatrix(p_V2, p_F2, igl::MASSMATRIX_TYPE_VORONOI, M);
    igl::invert_diag(M, Minv);

    // L, M = robust_laplacian.mesh_laplacian(V2, F2)
    Eigen::SparseMatrix<double> Q = -L + L * Minv * L;
    Eigen::SparseMatrix<double> Aeq;
    Eigen::VectorXd Beq;
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(L.rows(), p_W2.cols());

    Eigen::VectorXi b_all = Eigen::VectorXi::LinSpaced(p_V2.rows(), 0, p_V2.rows() - 1);
    Eigen::VectorXi b;
    igl::slice_mask(b_all, p_Matched, 1, b);

    Eigen::MatrixXd bc;
    igl::slice_mask(p_W2, p_Matched, 1, bc);

    igl::min_quad_with_fixed_data<double> mqwf;
    igl::min_quad_with_fixed_precompute(Q, b, Aeq, true, mqwf);

    bool result = igl::min_quad_with_fixed_solve(mqwf, B, bc, Beq, r_W_inpainted);

    return result;
}

/**
 * Smooth weights in the areas for which weights were inpainted and also their close neighbours.
 * 
 *  V2: #V2 by 3 target mesh vertices
 *  F2: #F2 by 3 target mesh triangles indices
 *  W2: #V2 by num_bones skinning weights
 *  Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
 *  dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
 *  num_smooth_iter_steps: scalar number of smoothing steps
 *  smooth_alpha: scalar the smoothing strength      
 *  W2_smoothed: #V2 by num_bones new smoothed weights
 *  VIDs_to_smooth: 1D array of vertex IDs for which smoothing was applied
 */
void smooth(Eigen::MatrixXd& W2_smoothed,
            Eigen::Array<bool,Eigen::Dynamic,1>& VIDs_to_smooth,
            const Eigen::MatrixXd& V2, 
            const Eigen::MatrixXi& F2, 
            const Eigen::MatrixXd& W2, 
            const Eigen::Array<bool,Eigen::Dynamic,1>& Matched, 
            const double dDISTANCE_THRESHOLD, 
            const double num_smooth_iter_steps, 
            const double smooth_alpha)
{
    Eigen::Array<bool,Eigen::Dynamic,1> NotMatched = Matched.select(Eigen::Array<bool,Eigen::Dynamic,1>::Constant(Matched.size(), false), Eigen::Array<bool,Eigen::Dynamic,1>::Constant(Matched.size(), true));
    VIDs_to_smooth = NotMatched; //.array(NotMatched, copy=True)

    std::vector<std::vector<int> > adj_list;
    igl::adjacency_list(F2, adj_list);

    auto get_points_within_distance = [&](const Eigen::MatrixXd& V, const int VID, const double distance)
    {
        // Get all neighbours of vertex VID within dDISTANCE_THRESHOLD   
        std::queue<int> queue;
        queue.push(VID);

        std::set<int> visited;
        visited.insert(VID);
        while (!queue.empty())
        {
            const int vv = queue.front();
            queue.pop();
            
            auto neigh = adj_list[vv];
            for (auto nn : neigh)
            {
                if (!VIDs_to_smooth[nn] && (V.row(VID) - V.row(nn)).norm() < distance)
                {
                    VIDs_to_smooth[nn] = true;
                    if (visited.find(nn) == visited.end())
                    {
                        queue.push(nn);
                        visited.insert(nn);
                    }
                }
            }
        }
    };

    for (int i = 0; i < V2.rows(); ++i)
    {
        if (NotMatched[i])
        {
            get_points_within_distance(V2, i, dDISTANCE_THRESHOLD);
        }
    }

    W2_smoothed = W2;

    for (int step_idx = 0;  step_idx < num_smooth_iter_steps; ++step_idx)
    {
        for (int i = 0; i < V2.rows(); ++i)
        {
            if (VIDs_to_smooth[i])
            {
                auto neigh = adj_list[i];
                int num = neigh.size();
                Eigen::VectorXd weight = W2_smoothed.row(i);

                Eigen::VectorXd new_weight = (1.0-smooth_alpha)*weight;

                for (auto influence_idx : neigh)
                {
                    Eigen::VectorXd weight_connected = W2_smoothed.row(influence_idx);
                    new_weight = new_weight + (smooth_alpha/num) * weight_connected;
                }
                
                W2_smoothed.row(i) = new_weight; 
            }
        }
    }
}

bool test_find_closest_point_on_surface() {
    Eigen::MatrixXd vertices(3, 3);
    vertices << -1, -1, -1,
                1, -1, -1,
                1, 1, -1;
    Eigen::MatrixXi triangles(1, 3);
    triangles << 0, 1, 2;
    Eigen::MatrixXd test_points(3, 3);
    test_points << 0, 0, 0,
                   2, 2, 2,
                   0, 0, -2;
    Eigen::MatrixXd expected_points(3, 3);
    expected_points << 0, 0, -1,
                       1, 1, -1,
                       0, 0, -1;

    Eigen::VectorXd sqrD;
    Eigen::VectorXi I;
    Eigen::MatrixXd C, B;
    find_closest_point_on_surface(test_points, vertices, triangles, sqrD, I, C, B);
    std::cout << "Closest Points:\n" << C << std::endl;
    std::cout << "Expected Points:\n" << expected_points << std::endl;
    if (!C.isApprox(expected_points, 1e-6)) {
        return false;
    }
    return true;
}

bool test_interpolate_attribute_from_bary() {
    Eigen::MatrixXd vertex_attributes(5, 2);
    vertex_attributes << 1, 2,
                         3, 4,
                         5, 6,
                         7, 8,
                         9, 10;
    Eigen::MatrixXd barycentric_coordinates(2, 3);
    barycentric_coordinates << 0.2, 0.5, 0.3,
                               0.6, 0.3, 0.1;
    Eigen::VectorXi primitive_indices(2);
    primitive_indices << 0, 1;
    Eigen::MatrixXi mesh_triangles(2, 3);
    mesh_triangles << 0, 1, 2,
                      2, 3, 4;
    Eigen::MatrixXd expected_output(2, 2);
    expected_output << 1 * 0.2 + 3 * 0.5 + 5 * 0.3, 2 * 0.2 + 4 * 0.5 + 6 * 0.3,
                       5 * 0.6 + 7 * 0.3 + 9 * 0.1, 6 * 0.6 + 8 * 0.3 + 10 * 0.1;

    Eigen::MatrixXd result;
    interpolate_attribute_from_bary(vertex_attributes, barycentric_coordinates, primitive_indices, mesh_triangles, result);
    std::cout << "Interpolated Attributes:\n" << result << std::endl;
    std::cout << "Expected Output:\n" << expected_output << std::endl;
    if (!result.isApprox(expected_output, 1e-6)) {
        return false;
    }
    return true;
}

bool test_normalize_vector() {
    Eigen::VectorXd vector(3);
    vector << 3, 4, 0;
    Eigen::VectorXd expected(3);
    expected << 0.6, 0.8, 0;

    Eigen::VectorXd normalized = normalize_vector(vector);
    std::cout << "Normalized Vector:\n" << normalized << std::endl;
    std::cout << "Expected Vector:\n" << expected << std::endl;
    if (!normalized.isApprox(expected, 1e-6)) {
        return false;
    }
    return true;
}

bool test_find_matches_closest_surface() {
    Eigen::MatrixXd source_vertices(3, 3);
    source_vertices << 0, 0, 0,
                       1, 0, 0,
                       0, 1, 0;
    Eigen::MatrixXi source_triangles(1, 3);
    source_triangles << 0, 1, 2;
    Eigen::MatrixXd source_normals(3, 3);
    source_normals << 0, 0, 1,
                      0, 0, 1,
                      0, 0, 1;
    Eigen::MatrixXd source_weights(3, 2);
    source_weights << 1, 0,
                      0, 1,
                      0.5, 0.5;

    Eigen::MatrixXd target_vertices(2, 3);
    target_vertices << 0.1, 0.1, 0,
                       2, 2, 2;
    Eigen::MatrixXi target_triangles(1, 3);
    target_triangles << 0, 1;
    Eigen::MatrixXd target_normals(2, 3);
    target_normals << 0, 0, 1,
                      1, 0, 0;

    double distance_threshold_squared = 0.5;
    double angle_threshold_degrees = 10;

    Eigen::VectorXi expected_matched(2);
    expected_matched << 1, 0;
    Eigen::MatrixXd expected_weights(2, 2);
    expected_weights << 0.85, 0.15,
                        0.25, 0.75;

    Eigen::MatrixXd target_weights;
    Eigen::Array<bool, Eigen::Dynamic, 1> matched;
    find_matches_closest_surface(source_vertices, source_triangles, source_normals, target_vertices, target_triangles, target_normals,
                                 source_weights, distance_threshold_squared, angle_threshold_degrees, target_weights, matched);
    std::cout << "Target Matched:\n" << matched << std::endl;
    std::cout << "Expected Matched:\n" << expected_matched << std::endl;
    std::cout << "Target Weights:\n" << target_weights << std::endl;
    std::cout << "Expected Weights:\n" << expected_weights << std::endl;
    if (!matched.cast<int>().isApprox(expected_matched.cast<int>()) || !target_weights.isApprox(expected_weights, 1e-6)) {
        return false;
    }
    return true;
}

bool test_is_valid_array() {
    Eigen::MatrixXd valid_matrix(2, 2);
    valid_matrix << 1, 2,
                    3, 4;
    Eigen::MatrixXd invalid_matrix(2, 2);
    invalid_matrix << std::numeric_limits<double>::quiet_NaN(), 2,
                      std::numeric_limits<double>::infinity(), 4;

    std::cout << "Valid Matrix:\n" << valid_matrix << std::endl;
    std::cout << "Invalid Matrix:\n" << invalid_matrix << std::endl;
    if (is_valid_array(valid_matrix) != true || is_valid_array(invalid_matrix) != false) {
        return false;
    }
    return true;
}

bool test_inpaint() {
    Eigen::MatrixXd V2(4, 3);
    V2 << 0, 0, 0,
          1, 0, 0,
          0, 1, 0,
          1, 1, 0;
    Eigen::MatrixXi F2(2, 3);
    F2 << 0, 1, 2,
          1, 2, 3;
    Eigen::MatrixXd W2(4, 2);
    W2 << 1, 0,
          0, 1,
          0.5, 0.5,
          0, 0;
    Eigen::Array<bool, Eigen::Dynamic, 1> Matched(4);
    Matched << true, true, true, false;
    Eigen::MatrixXd expected_W_inpainted(4, 2);
    expected_W_inpainted << 1, 0,
                            0, 1,
                            0.5, 0.5,
                            0.0357143 , 0.964286;

    Eigen::MatrixXd W_inpainted(4, 2);
    
    bool success = inpaint(V2, F2, W2, Matched, W_inpainted);
    std::cout << "Inpainted Weights:\n" << W_inpainted << std::endl;
    std::cout << "Expected Inpainted Weights:\n" << expected_W_inpainted << std::endl;
    if (success != true || !W_inpainted.isApprox(expected_W_inpainted, 1e-6)) {
        return false;
    }
    return true;
}

bool test_smooth() {
    Eigen::MatrixXd target_vertices(5, 3);
    target_vertices << 0, 0, 0,
                       1, 0, 0,
                       0, 1, 0,
                       1, 1, 0,
                       2, 1, 0;
    Eigen::MatrixXi target_faces(2, 3);
    target_faces << 0, 1, 2,
                    1, 2, 3;
    Eigen::MatrixXd skinning_weights(5, 2);
    skinning_weights << 1, 0,
                        0, 1,
                        0.5, 0.5,
                        0.25, 0.75,
                        0.1, 0.9;
    Eigen::Array<bool, Eigen::Dynamic, 1> matched(5);
    matched << true, true, true, false, false;
    double distance_threshold = 1.5;

    Eigen::MatrixXd expected_smoothed_weights(5, 2);
    expected_smoothed_weights << 0.85, 0.15,
                                 0.106667, 0.893333,
                                 0.480444, 0.519556,
                                 0.258711, 0.741289,
                                 0.08, 0.72;
    Eigen::Array<bool, Eigen::Dynamic, 1> expected_vertices_ids_to_smooth(5);
    expected_vertices_ids_to_smooth << true, true, true, true, true;

    Eigen::MatrixXd smoothed_weights;
    Eigen::Array<bool, Eigen::Dynamic, 1> vertices_ids_to_smooth;
    smooth(smoothed_weights, vertices_ids_to_smooth, target_vertices, target_faces, skinning_weights, matched, distance_threshold, 1, 0.2);

    std::cout << "Smoothed Weights:\n" << smoothed_weights << std::endl;
    std::cout << "Expected Smoothed Weights:\n" << expected_smoothed_weights << std::endl;
    std::cout << "Vertices IDs to Smooth:\n" << vertices_ids_to_smooth << std::endl;
    std::cout << "Expected Vertices IDs to Smooth:\n" << expected_vertices_ids_to_smooth << std::endl;

    if (!smoothed_weights.isApprox(expected_smoothed_weights, 1e-6) || (vertices_ids_to_smooth != expected_vertices_ids_to_smooth).any()) {
        return false;
    }
    return true;
}

bool test_find_matches_closest_surface_mesh() {
    Eigen::MatrixXd source_vertices(4, 3);
    source_vertices << 0, 0, 0,
                       1, 0, 0,
                       0, 1, 0,
                       1, 1, 0;
    Eigen::MatrixXi source_triangles(2, 3);
    source_triangles << 0, 1, 2,
                        1, 2, 3;
    Eigen::MatrixXd source_normals(4, 3);
    source_normals << 0, 0, 1,
                      0, 0, 1,
                      0, 0, 1,
                      0, 0, 1;
    Eigen::MatrixXd source_weights(4, 8);
    source_weights << 1, 0, 0, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 0, 0,
                      0.5, 0.5, 0, 0, 0, 0, 0, 0,
                      0.25, 0.75, 0, 0, 0, 0, 0, 0;

    Eigen::MatrixXd target_vertices(3, 3);
    target_vertices << 0.5, 0.5, 0,
                       1.5, 1.5, 0,
                       0.5, 0.5, 1;
    Eigen::MatrixXi target_triangles(1, 3);
    target_triangles << 0, 1, 2;
    Eigen::MatrixXd target_normals(3, 3);
    target_normals << 0, 0, 1,
                      0, 0, 1,
                      0, 0, 1;

    double distance_threshold_squared = 1.0;
    double angle_threshold_degrees = 10;

    Eigen::Array<bool, Eigen::Dynamic, 1> expected_matched(3);
    expected_matched << true, true, true;
    Eigen::MatrixXd expected_weights(3, 8);
    expected_weights << 0.25, 0.75, 0, 0, 0, 0, 0, 0,
                        0.25, 0.75, 0, 0, 0, 0, 0, 0,
                        0.25, 0.75, 0, 0, 0, 0, 0, 0;

    Eigen::MatrixXd target_weights;
    Eigen::Array<bool, Eigen::Dynamic, 1> matched;
    find_matches_closest_surface(source_vertices, source_triangles, source_normals, target_vertices, target_triangles, target_normals,
                                 source_weights, distance_threshold_squared, angle_threshold_degrees, target_weights, matched);
    std::cout << "Target Matched:\n" << matched << std::endl;
    std::cout << "Expected Matched:\n" << expected_matched << std::endl;
    std::cout << "Target Weights:\n" << target_weights << std::endl;
    std::cout << "Expected Weights:\n" << expected_weights << std::endl;
    if (!matched.isApprox(expected_matched) || !target_weights.isApprox(expected_weights, 1e-6)) {
        return false;
    }
    return true;
}

bool test_find_matches_closest_surface_no_weights() {
    Eigen::MatrixXd source_vertices(3, 3);
    source_vertices << 0, 0, 0,
                       1, 0, 0,
                       0, 1, 0;
    Eigen::MatrixXi source_triangles(1, 3);
    source_triangles << 0, 1, 2;
    Eigen::MatrixXd source_normals(3, 3);
    source_normals << 0, 0, 1,
                      0, 0, 1,
                      0, 0, 1;

    Eigen::MatrixXd target_vertices(2, 3);
    target_vertices << 0.1, 0.1, 0,
                       2, 2, 2;
    Eigen::MatrixXi target_triangles(1, 3);
    target_triangles << 0, 1;
    Eigen::MatrixXd target_normals(2, 3);
    target_normals << 0, 0, 1,
                      1, 0, 0;

    double distance_threshold_squared = 0.5;
    double angle_threshold_degrees = 10;

    Eigen::VectorXi expected_matched(2);
    expected_matched << 1, 0;

    Eigen::MatrixXd target_weights;
    Eigen::Array<bool, Eigen::Dynamic, 1> matched;
    find_matches_closest_surface(source_vertices, source_triangles, source_normals, target_vertices, target_triangles, target_normals,
                                 Eigen::MatrixXd(), distance_threshold_squared, angle_threshold_degrees, target_weights, matched);
    std::cout << "Target Matched:\n" << matched << std::endl;
    std::cout << "Expected Matched:\n" << expected_matched << std::endl;
    if (!matched.cast<int>().isApprox(expected_matched.cast<int>())) {
        return false;
    }
    return true;
}

extern "C" Variant robust_weight_transfer(Mesh source_mesh, Mesh target_mesh, Dictionary arguments, Array matched_array, Array interpolated_weights_array, Array inpainted_weights_array, Array smoothed_weights_array) {
    if (!arguments.has("verbose") || !arguments.has("angle_threshold_degrees") || !arguments.has("distance_threshold") || !arguments.has("source_mesh_surface") || !arguments.has("target_mesh_surface")) {
        std::cerr << "Missing required arguments" << std::endl;
        return false;
    }

    bool verbose = arguments["verbose"].value();
    double angle_threshold_degrees = arguments["angle_threshold_degrees"].value();
    double distance_threshold = arguments["distance_threshold"].value();
    int64_t source_mesh_surface = arguments["source_mesh_surface"].value();
    int64_t target_mesh_surface = arguments["target_mesh_surface"].value();

    if (verbose) { std::cout << "arguments:\n" << std::endl; }

    Array source_mesh_arrays = source_mesh.surface_get_arrays(source_mesh_surface);
    if (source_mesh_arrays.size() <= Mesh::ARRAY_VERTEX || source_mesh_arrays.size() <= Mesh::ARRAY_INDEX || source_mesh_arrays.size() <= Mesh::ARRAY_NORMAL || source_mesh_arrays.size() <= Mesh::ARRAY_WEIGHTS) {
        std::cerr << "Source mesh arrays are incomplete" << std::endl;
        return false;
    }

    Variant vertices_1_variant = source_mesh_arrays[Mesh::ARRAY_VERTEX].get();
    if (vertices_1_variant == Nil) {
        std::cerr << "vertices_1_variant is null" << std::endl;
        return false;
    }
    PackedArray<Vector3> vertices_1_ref = vertices_1_variant;

    Variant faces_1_variant = source_mesh_arrays[Mesh::ARRAY_INDEX].get();
    if (faces_1_variant == Nil) {
        std::cerr << "faces_1_variant is null" << std::endl;
        return false;
    }
    PackedArray<int32_t> faces_1_ref = faces_1_variant;

    Variant normals_1_variant = source_mesh_arrays[Mesh::ARRAY_NORMAL].get();
    if (normals_1_variant == Nil) {
        std::cerr << "normals_1_variant is null" << std::endl;
        return false;
    }
    PackedArray<Vector3> normals_1_ref = normals_1_variant;

    Variant skin_weights_variant = source_mesh_arrays[Mesh::ARRAY_WEIGHTS].get();
    if (skin_weights_variant == Nil) {
        std::cerr << "skin_weights_variant is null" << std::endl;
        return false;
    }
    PackedArray<float> skin_weights_ref = skin_weights_variant;

    if (verbose) { std::cout << "source_mesh_arrays:\n" << std::endl; }

    if (vertices_1_ref.is_empty() || faces_1_ref.is_empty() || normals_1_ref.is_empty() || skin_weights_ref.is_empty()) {
        std::cerr << "One or more source mesh arrays are empty" << std::endl;
        return false;
    }

    std::vector<Vector3> vertices_1 = vertices_1_ref.fetch();
    std::vector<int32_t> faces_1 = faces_1_ref.fetch();
    std::vector<Vector3> normals_1 = normals_1_ref.fetch();
    std::vector<float> skin_weights_1 = skin_weights_ref.fetch();

    Array target_mesh_arrays = target_mesh.surface_get_arrays(target_mesh_surface);
    if (target_mesh_arrays.size() <= Mesh::ARRAY_VERTEX || target_mesh_arrays.size() <= Mesh::ARRAY_INDEX || target_mesh_arrays.size() <= Mesh::ARRAY_NORMAL) {
        std::cerr << "Target mesh arrays are incomplete" << std::endl;
        return false;
    }
    Variant vertices_2_variant = target_mesh_arrays[Mesh::ARRAY_VERTEX].get();
    if (vertices_2_variant == Nil) {
        std::cerr << "vertices_2_variant is null" << std::endl;
        return false;
    }
    PackedArray<Vector3> vertices_2_ref = vertices_2_variant;

    Variant faces_2_variant = target_mesh_arrays[Mesh::ARRAY_INDEX].get();
    if (faces_2_variant == Nil) {
        std::cerr << "faces_2_variant is null" << std::endl;
        return false;
    }
    PackedArray<int32_t> faces_2_ref = faces_2_variant;

    Variant normals_2_variant = target_mesh_arrays[Mesh::ARRAY_NORMAL].get();
    if (normals_2_variant == Nil) {
        std::cerr << "normals_2_variant is null" << std::endl;
        return false;
    }
    PackedArray<Vector3> normals_2_ref = normals_2_variant;

    if (verbose) { std::cout << "target_mesh_arrays:\n" << std::endl; }

    if (vertices_2_ref.is_empty() || faces_2_ref.is_empty() || normals_2_ref.is_empty()) {
        std::cerr << "One or more target mesh arrays are empty" << std::endl;
        return false;
    }

    std::vector<Vector3> vertices_2 = vertices_2_ref.fetch();
    std::vector<int32_t> faces_2 = faces_2_ref.fetch();
    std::vector<Vector3> normals_2 = normals_2_ref.fetch();

    Eigen::MatrixXd vertices_1_eigen(vertices_1.size(), 3);
    for (int i = 0; i < vertices_1.size(); ++i) {
        Vector3 v = vertices_1[i];
        vertices_1_eigen(i, 0) = v.x;
        vertices_1_eigen(i, 1) = v.y;
        vertices_1_eigen(i, 2) = v.z;
    }
    if (verbose) { std::cout << "vertices_1_eigen:\n" << vertices_1_eigen << std::endl; }

    Eigen::MatrixXi faces_1_eigen(faces_1.size() / 3, 3);
    for (int i = 0; i < faces_1.size() / 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            faces_1_eigen(i, j) = faces_1[i * 3 + j];
        }
    }
    if (verbose) { std::cout << "faces_1_eigen:\n" << faces_1_eigen << std::endl; }

    Eigen::MatrixXd normals_1_eigen(normals_1.size(), 3);
    for (int i = 0; i < normals_1.size(); ++i) {
        Vector3 n = normals_1[i];
        normals_1_eigen(i, 0) = n.x;
        normals_1_eigen(i, 1) = n.y;
        normals_1_eigen(i, 2) = n.z;
    }
    if (verbose) { std::cout << "normals_1_eigen:\n" << normals_1_eigen << std::endl; }

    if (skin_weights_1.empty()) {
        std::cerr << "skin_weights array is empty" << std::endl;
        return false;
    }
    int32_t num_bones = skin_weights_1.size() / vertices_1.size();
    Eigen::MatrixXd skin_weights_eigen(vertices_1.size(), num_bones);
    for (int i = 0; i < vertices_1.size(); ++i) {
        for (int j = 0; j < num_bones; ++j) {
            skin_weights_eigen(i, j) = skin_weights_1[i * num_bones + j];
        }
    }

    if (verbose) { std::cout << "skin_weights_eigen:\n" << skin_weights_eigen << std::endl; }

    Eigen::MatrixXd vertices_2_eigen(vertices_2.size(), 3);
    for (int i = 0; i < vertices_2.size(); ++i) {
        Vector3 v = vertices_2[i];
        vertices_2_eigen(i, 0) = v.x;
        vertices_2_eigen(i, 1) = v.y;
        vertices_2_eigen(i, 2) = v.z;
    }
    if (verbose) { std::cout << "vertices_2_eigen:\n" << vertices_2_eigen << std::endl; }

    Eigen::MatrixXi faces_2_eigen(faces_2.size() / 3, 3);
    for (int i = 0; i < faces_2.size() / 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            faces_2_eigen(i, j) = faces_2[i * 3 + j];
        }
    }
    if (verbose) { std::cout << "faces_2_eigen:\n" << faces_2_eigen << std::endl; }

    Eigen::MatrixXd normals_2_eigen(normals_2.size(), 3);
    for (int i = 0; i < normals_2.size(); ++i) {
        Vector3 n = normals_2[i];
        normals_2_eigen(i, 0) = n.x;
        normals_2_eigen(i, 1) = n.y;
        normals_2_eigen(i, 2) = n.z;
    }
    if (verbose) { std::cout << "normals_2_eigen:\n" << normals_2_eigen << std::endl; }

    // Section 3.1 Closest Point Matching
    if (verbose) { std::cout << "Distance threshold: " << distance_threshold << std::endl; }

    Eigen::MatrixXd W2_eigen = Eigen::MatrixXd::Zero(vertices_2_eigen.rows(), skin_weights_eigen.cols());
    Eigen::Array<bool, Eigen::Dynamic, 1> Matched_eigen;

    find_matches_closest_surface(vertices_1_eigen, faces_1_eigen, normals_1_eigen, vertices_2_eigen, faces_2_eigen, normals_2_eigen, skin_weights_eigen, distance_threshold * distance_threshold, angle_threshold_degrees, W2_eigen, Matched_eigen);
    if (verbose) { std::cout << "Matched_eigen:\n" << Matched_eigen << std::endl; }
    if (verbose) { std::cout << "W2_eigen:\n" << W2_eigen << std::endl; }

    // Section 3.2 Skinning Weights Inpainting
    Eigen::MatrixXd W_inpainted;
    bool success = inpaint(vertices_2_eigen, faces_2_eigen, W2_eigen, Matched_eigen, W_inpainted);
    if (verbose) { std::cout << "Inpainting success: " << success << std::endl; }
    if (verbose) { std::cout << "W_inpainted:\n" << W_inpainted << std::endl; }

    // Optional smoothing
    Eigen::MatrixXd W2_smoothed;
    Eigen::Array<bool, Eigen::Dynamic, 1> VIDs_to_smooth;
    smooth(W2_smoothed, VIDs_to_smooth, vertices_2_eigen, faces_2_eigen, W_inpainted, Matched_eigen, distance_threshold, 10, 0.2);
    if (verbose) { std::cout << "W2_smoothed:\n" << W2_smoothed << std::endl; }
    if (verbose) { std::cout << "VIDs_to_smooth:\n" << VIDs_to_smooth << std::endl; }
    
    if (verbose) { std::cout << "Matched: " << Matched_eigen.transpose() << std::endl; }
    if (verbose) { std::cout << "Interpolated Skin Weights: " << W2_eigen << std::endl; }
    if (verbose) { std::cout << "Inpainted Weights: " << W_inpainted << std::endl; }
    if (verbose) { std::cout << "Smoothed Inpainted Weights: " << W2_smoothed << std::endl; }

    for (int i = 0; i < Matched_eigen.size(); ++i) {
        matched_array.append(Matched_eigen(i));
    }
    if (verbose) { std::cout << "Matched array appended." << std::endl; }
    for (int i = 0; i < W2_eigen.size(); ++i) {
        interpolated_weights_array.append(Variant(W2_eigen(i)));
    }
    if (verbose) { std::cout << "Interpolated weights array appended." << std::endl; }

    for (int i = 0; i < W_inpainted.size(); ++i) {
        inpainted_weights_array.append(Variant(W_inpainted(i)));
    }
    if (verbose) { std::cout << "Inpainted weights array appended." << std::endl; }

    for (int i = 0; i < W2_smoothed.size(); ++i) {
        smoothed_weights_array.append(Variant(W2_smoothed(i)));
    }
    if (verbose) { std::cout << "Smoothed weights array appended." << std::endl; }

    return true;
}

bool test_robust_weight_transfer() {
    Eigen::MatrixXd vertices_1(3, 3);
    vertices_1 << 0, 0, 0,
                  1, 0, 0,
                  0, 1, 0;
    Eigen::MatrixXi faces_1(1, 3);
    faces_1 << 0, 1, 2;
    Eigen::MatrixXd normals_1(3, 3);
    normals_1 << 0, 0, 1,
                 0, 0, 1,
                 0, 0, 1;
    Eigen::MatrixXd skin_weights(3, 2);
    skin_weights << 1, 0,
                    0, 1,
                    0.5, 0.5;

    Eigen::MatrixXd vertices_2(2, 3);
    vertices_2 << 0.1, 0.1, 0,
                  2, 2, 2;
    Eigen::MatrixXi faces_2(1, 2);
    faces_2 << 0, 1;
    Eigen::MatrixXd normals_2(2, 3);
    normals_2 << 0, 0, 1,
                 1, 0, 0;

    // Section 3.1 Closest Point Matching
    double distance_threshold = 0.05 * (vertices_2.row(1) - vertices_2.row(0)).norm();
    double distance_threshold_squared = distance_threshold * distance_threshold;
    int angle_threshold_degrees = 30;

    Eigen::MatrixXd W2 = Eigen::MatrixXd::Zero(vertices_2.rows(), skin_weights.cols());
    Eigen::Array<bool, Eigen::Dynamic, 1> Matched;

    find_matches_closest_surface(vertices_1, faces_1, normals_1, vertices_2, faces_2, normals_2, skin_weights, distance_threshold_squared, angle_threshold_degrees, W2, Matched);

    // Section 3.2 Skinning Weights Inpainting
    Eigen::MatrixXd W_inpainted;
    bool success = inpaint(vertices_2, faces_2, W2, Matched, W_inpainted);
    std::cout << "Inpainting success: " << success << std::endl;

    // Optional smoothing
    Eigen::MatrixXd W2_smoothed;
    Eigen::Array<bool, Eigen::Dynamic, 1> VIDs_to_smooth;
    smooth(W2_smoothed, VIDs_to_smooth, vertices_2, faces_2, W_inpainted, Matched, distance_threshold, 10, 0.2);

    // Print results
    std::cout << "Matched: " << Matched.transpose() << std::endl;
    std::cout << "Interpolated Skin Weights: " << W2 << std::endl;
    std::cout << "Inpainted Weights: " << W_inpainted << std::endl;
    std::cout << "Smoothed Inpainted Weights: " << W2_smoothed << std::endl;
    return true;
}

extern "C" Variant run_tests() {
    bool all_tests_passed = true;
    if (!test_find_closest_point_on_surface()) {
        std::cerr << "test_find_closest_point_on_surface failed" << std::endl;
        all_tests_passed = false;
    }
    if (!test_interpolate_attribute_from_bary()) {
        std::cerr << "test_interpolate_attribute_from_bary failed" << std::endl;
        all_tests_passed = false;
    }
    if (!test_normalize_vector()) {
        std::cerr << "test_normalize_vector failed" << std::endl;
        all_tests_passed = false;
    }
    if (!test_find_matches_closest_surface()) {
        std::cerr << "test_find_matches_closest_surface failed" << std::endl;
        all_tests_passed = false;
    }
    if (!test_find_matches_closest_surface_no_weights()) {
        std::cerr << "test_find_matches_closest_surface_no_weights failed" << std::endl;
        all_tests_passed = false;
    }
    if (!test_is_valid_array()) {
        std::cerr << "test_is_valid_array failed" << std::endl;
        all_tests_passed = false;
    }
    if (!test_smooth()) {
        std::cerr << "test_smooth failed" << std::endl;
        all_tests_passed = false;
    }
    if (!test_inpaint()) {
        std::cerr << "test_inpaint failed" << std::endl;
        all_tests_passed = false;
    }
    if (!test_find_matches_closest_surface_mesh()) {
        std::cerr << "test_find_matches_closest_surface_mesh failed" << std::endl;
        all_tests_passed = false;
    }
    if (!test_robust_weight_transfer()) {
        std::cerr << "test_robust_weight_transfer failed" << std::endl;
        all_tests_passed = false;
    }
    if (all_tests_passed) {
        std::cout << "All tests passed!" << std::endl;
        return Variant(0);
    } else {
        std::cerr << "Some tests failed." << std::endl;
        return Variant(1);
    }
}