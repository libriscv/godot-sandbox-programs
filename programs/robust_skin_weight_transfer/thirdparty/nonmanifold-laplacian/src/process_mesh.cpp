#include "bubble_offset.h"
#include "point_cloud_utilities.h"
#include "utils.hpp"

#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/halfedge_factories.h"
#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/signpost_intrinsic_triangulation.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/tufted_laplacian.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <sstream>

using namespace geometrycentral;
using namespace geometrycentral::surface;

// core data
std::unique_ptr<SurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

// Parameters
float mollifyFactor = 0.;
bool isPointCloud = false;
unsigned int nNeigh = 30;
double laplacianReplaceVal = 1.;
double massReplaceVal = -1e-3;

template <typename T>
void saveMatrix(std::string filename, SparseMatrix<T>& matrix) {

  // WARNING: this follows matlab convention and thus is 1-indexed

  std::cout << "Writing sparse matrix to: " << filename << std::endl;

  std::ofstream outFile(filename);
  if (!outFile) {
    throw std::runtime_error("failed to open output file " + filename);
  }

  // Write a comment on the first line giving the dimensions
  // outFile << "# sparse " << matrix.rows() << " " << matrix.cols() << std::endl;

  outFile << std::setprecision(16);

  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (typename SparseMatrix<T>::InnerIterator it(matrix, k); it; ++it) {
      T val = it.value();
      size_t iRow = it.row();
      size_t iCol = it.col();

      outFile << (iRow + 1) << " " << (iCol + 1) << " " << val << std::endl;
    }
  }

  outFile.close();
}

void processMesh(const std::string& inputFilename, float mollifyFactor, unsigned int nNeigh, double laplacianReplaceVal, double massReplaceVal, const std::string& outputPrefix, bool gui, bool writeLaplacian, bool writeMass) {
  // Make sure a mesh name was given
  if (inputFilename.empty()) {
    return;
  }

  // Load mesh
  SimplePolygonMesh inputMesh(inputFilename);

  // if it's a point cloud, generate some triangles
  isPointCloud = inputMesh.polygons.empty();
  if (isPointCloud) {
    std::cout << "Detected point cloud input" << std::endl;
    Neighbors_t neigh = generate_knn(inputMesh.vertexCoordinates, nNeigh);
    std::vector<Vector3> normals = generate_normals(inputMesh.vertexCoordinates, neigh);
    std::vector<std::vector<Vector2>> coords = generate_coords_projection(inputMesh.vertexCoordinates, normals, neigh);
    LocalTriangulationResult localTri = build_delaunay_triangulations(coords, neigh);

    // Take the union of all triangles in all the neighborhoods
    for (size_t iPt = 0; iPt < inputMesh.vertexCoordinates.size(); iPt++) {
      const std::vector<size_t>& thisNeigh = neigh[iPt];
      size_t nNeigh = thisNeigh.size();

      // Accumulate over triangles
      for (const auto& tri : localTri.pointTriangles[iPt]) {
        std::array<size_t, 3> triGlobal = {iPt, thisNeigh[tri[1]], thisNeigh[tri[2]]};
        inputMesh.polygons.push_back({triGlobal[0], triGlobal[1], triGlobal[2]});
      }
    }
  }

  // make sure the input really is a triangle mesh
  inputMesh.stripFacesWithDuplicateVertices(); // need a richer format to encode these
  std::vector<size_t> oldToNewMap = inputMesh.stripUnusedVertices();
  inputMesh.triangulate(); // probably what the user wants

  std::tie(mesh, geometry) = makeGeneralHalfedgeAndGeometry(inputMesh.polygons, inputMesh.vertexCoordinates);


  // ta-da! (invoke the algorithm from geometry-central)
  std::cout << "Building tufted Laplacian..." << std::endl;
  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);
  if (isPointCloud) {
    L = L / 3.;
    M = M / 3.;
  }
  std::cout << "  ...done!" << std::endl;

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {

    // Invert the map
    std::vector<size_t> newToOldMap(inputMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }

  // write output matrices, if requested
  if (writeLaplacian) {
    saveMatrix(outputPrefix + "laplacian.spmat", L);
  }
  if (writeMass) {
    saveMatrix(outputPrefix + "lumped_mass.spmat", M);
  }
}
