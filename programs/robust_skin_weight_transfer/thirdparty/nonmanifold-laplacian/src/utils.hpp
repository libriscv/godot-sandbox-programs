#include "bubble_offset.h"

#include "geometrycentral/surface/halfedge_factories.h"
#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/simple_idt.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/timing.h"

namespace {

// Return indices from sort
template <typename T>
std::vector<size_t> sortedInds(const std::vector<T>& v) {
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&v](size_t i, size_t j) { return v[i] < v[j]; });
  return idx;
}

} // namespace

VertexData<Vertex> splitVertexNeighborhoods(HalfedgeMesh& mesh, VertexPositionGeometry& geom) {

  /*
  geom.requireFaceNormals(); // for debugging

  // Idenfity distinct neighborhoods
  VertexData<size_t> sourceInds = mesh.getVertexIndices();
  CornerData<size_t> neighInd(mesh, INVALID_IND);
  VertexData<size_t> neighCount(mesh, 0);
  size_t nNeigh = 0;
  std::vector<size_t> newToOldMap;
  std::vector<Vector3> neighCloud;
  for (Halfedge he : mesh.halfedges()) {
    if (!he.isInterior()) throw std::runtime_error("mesh should be closed");
    if (neighInd[he.corner()] != INVALID_IND) continue;

    size_t thisNeigh = nNeigh++;
    newToOldMap.push_back(sourceInds[he.vertex()]);
    neighCount[he.vertex()]++;
    Vector3 origPos = geom.inputVertexPositions[he.vertex()];
    Vector3 normal = Vector3::zero();

    Halfedge startHe = he;
    Halfedge currHe = startHe;
    do {
      if (neighInd[currHe.corner()] != INVALID_IND)
        throw std::runtime_error("part of neighborhood has already been visited");
      neighInd[currHe.corner()] = thisNeigh;
      normal += geom.faceNormals[currHe.face()];
      currHe = currHe.twin().next();
    } while (currHe != startHe);

    if(norm(normal) > 1e-6) {
      normal = unit(normal);
    }
    neighCloud.push_back(origPos + 0.05 * normal);
  }

  polyscope::getSurfaceMesh("orig input")->addVertexScalarQuantity("neigh count", neighCount);
  polyscope::registerPointCloud("neigh points", neighCloud);
  std::vector<size_t> neighCloudInds(neighCloud.size());
  std::iota(neighCloudInds.begin(), neighCloudInds.end(), 0);
  polyscope::getPointCloud("neigh points")->addScalarQuantity("ind", neighCloudInds);
  polyscope::show();

  // The output is really a manifold mesh, so we can just build a triangle list
  std::vector<std::vector<size_t>> polygons;
  for (Face f : mesh.faces()) {

    size_t neighI = neighInd[f.halfedge().corner()];
    size_t neighJ = neighInd[f.halfedge().next().corner()];
    size_t neighK = neighInd[f.halfedge().next().next().corner()];

    polygons.push_back({neighI, neighJ, neighK});
  }

  std::unique_ptr<HalfedgeMesh> newMesh(new HalfedgeMesh(polygons));

  // Populate the original indices
  VertexData<size_t> origInds(*newMesh);
  for (size_t iV = 0; iV < newMesh->nVertices(); iV++) {
    origInds[iV] = newToOldMap[iV];
  }
  */

  VertexData<Vertex> origVerts = mesh.separateNonmanifoldVertices();

  return origVerts;
}

void applySplit(VertexPositionGeometry& geom, const VertexData<Vertex>& origVert) {
  SurfaceMesh& mesh = geom.mesh;
  for (Vertex v : mesh.vertices()) {
    geom.inputVertexPositions[v] = geom.inputVertexPositions[origVert[v]];
  }
}

std::unique_ptr<PolygonSoupMesh> subdivideRounded(VertexPositionGeometry& geom, int subdivLevel, double scale,
                                                  double dialate, double normalOffset) {

  // Make a copy
  SurfaceMesh* mesh = &geom.mesh;
  geom.requireVertexPositions();
  geom.requireFaceNormals();
  geom.requireFaceAreas();
  geom.requireEdgeLengths();


  // == Good old-fashioned subdivision, preserving barycentric coords on to original triangle
  std::vector<std::array<size_t, 3>> faces;       // per-face
  std::vector<Vector3> vertexCoords;              // per-vert
  std::vector<std::array<Vector3, 3>> baryCoords; // per-face
  std::vector<Face> origFaceTri;                  // per-face

  { // Initialize
    for (Face f : mesh->faces()) {
      baryCoords.push_back({Vector3{1., 0., 0.}, Vector3{0., 1., 0.}, Vector3{0., 0., 1.}});
      origFaceTri.push_back(f);

      faces.push_back({vertexCoords.size(), vertexCoords.size() + 1, vertexCoords.size() + 2});

      // create distinct copies of the vertices
      vertexCoords.push_back(geom.vertexPositions[f.halfedge().vertex()]);
      vertexCoords.push_back(geom.vertexPositions[f.halfedge().next().vertex()]);
      vertexCoords.push_back(geom.vertexPositions[f.halfedge().next().next().vertex()]);
    }
  }

  // Iteratively subdivide
  for (int iSub = 0; iSub < subdivLevel; iSub++) {

    // New arrays
    std::vector<Vector3> vertexCoordsNew = vertexCoords; // per-vertex, copy existing
    // std::vector<Vector3> vertexOrigBaryCoordsNew = vertexOrigBaryCoords; // per-vertex, copy existing
    std::vector<std::array<size_t, 3>> facesNew;       // per-face, all new
    std::vector<Face> origFaceTriNew;                  // per-face, all new
    std::vector<std::array<Vector3, 3>> baryCoordsNew; // per-face, all new

    // Create new vertices along edges
    std::map<std::tuple<size_t, size_t>, size_t> edgeVert;
    std::vector<std::array<size_t, 3>> faceVert; // used if iSub == 0
    if (iSub == 0) {
      faceVert.resize(faces.size());
      // if (false) {
      // On the first iteration, separate the edges
      for (size_t iF = 0; iF < faces.size(); iF++) {
        for (size_t j = 0; j < 3; j++) {
          size_t vTail = faces[iF][j];
          size_t vTip = faces[iF][(j + 1) % 3];

          size_t newInd = vertexCoordsNew.size();
          Vector3 newPos = (vertexCoords[vTail] + vertexCoords[vTip]) / 2.;
          vertexCoordsNew.push_back(newPos);

          faceVert[iF][j] = newInd;
        }
      }
    } else {

      for (size_t iF = 0; iF < faces.size(); iF++) {
        for (size_t j = 0; j < 3; j++) {
          size_t vTail = faces[iF][j];
          size_t vTip = faces[iF][(j + 1) % 3];

          size_t newInd = vertexCoordsNew.size();
          Vector3 newPos = (vertexCoords[vTail] + vertexCoords[vTip]) / 2.;
          vertexCoordsNew.push_back(newPos);

          std::tuple<size_t, size_t> key{vTail, vTip};
          std::tuple<size_t, size_t> keyTwin{vTip, vTail};
          if (edgeVert.find(key) == edgeVert.end()) {
            edgeVert[key] = newInd;
            edgeVert[keyTwin] = newInd;
          }
        }
      }
    }


    // Create new faces
    for (size_t iF = 0; iF < faces.size(); iF++) {

      // Gather vertices
      size_t vA = faces[iF][0];
      size_t vB = faces[iF][1];
      size_t vC = faces[iF][2];

      size_t vAB, vBC, vCA;
      if (iSub == 0) {
        vAB = faceVert[iF][0];
        vBC = faceVert[iF][1];
        vCA = faceVert[iF][2];
      } else {

        if (edgeVert.find(std::tuple<size_t, size_t>{vA, vB}) == edgeVert.end())
          throw std::runtime_error("edge key " + std::to_string(vA) + " --- " + std::to_string(vB));
        if (edgeVert.find(std::tuple<size_t, size_t>{vB, vC}) == edgeVert.end())
          throw std::runtime_error("edge key " + std::to_string(vB) + " --- " + std::to_string(vC));
        if (edgeVert.find(std::tuple<size_t, size_t>{vC, vA}) == edgeVert.end())
          throw std::runtime_error("edge key " + std::to_string(vC) + " --- " + std::to_string(vA));


        vAB = edgeVert[std::tuple<size_t, size_t>{vA, vB}];
        vBC = edgeVert[std::tuple<size_t, size_t>{vB, vC}];
        vCA = edgeVert[std::tuple<size_t, size_t>{vC, vA}];
      }

      if (vAB == 0 || vBC == 0 || vCA == 0) {
        std::cout << "  face " << iF << " has verts " << vAB << " " << vBC << " " << vCA << std::endl;
      }

      Vector3 bA = baryCoords[iF][0];
      Vector3 bB = baryCoords[iF][1];
      Vector3 bC = baryCoords[iF][2];
      Vector3 bAB = 0.5 * (bA + bB);
      Vector3 bBC = 0.5 * (bB + bC);
      Vector3 bCA = 0.5 * (bC + bA);

      // Create new faces

      facesNew.push_back({vA, vAB, vCA});
      baryCoordsNew.push_back({bA, bAB, bCA});
      origFaceTriNew.push_back(origFaceTri[iF]);

      facesNew.push_back({vAB, vB, vBC});
      baryCoordsNew.push_back({bAB, bB, bBC});
      origFaceTriNew.push_back(origFaceTri[iF]);

      facesNew.push_back({vCA, vBC, vC});
      baryCoordsNew.push_back({bCA, bBC, bC});
      origFaceTriNew.push_back(origFaceTri[iF]);

      facesNew.push_back({vAB, vBC, vCA});
      baryCoordsNew.push_back({bAB, bBC, bCA});
      origFaceTriNew.push_back(origFaceTri[iF]);
    }

    // Swap in new arrays
    vertexCoords = vertexCoordsNew;
    faces = facesNew;
    origFaceTri = origFaceTriNew;
    baryCoords = baryCoordsNew;
  }

  BubbleOffset bubbleOffset(geom);
  bubbleOffset.relativeScale = scale;
  bubbleOffset.dialate = dialate;
  bubbleOffset.normalOffset = normalOffset;

  // Apply offsets
  // (this will process each vertex many times; that's fine)
  std::vector<Vector3> vertexCoordsOffset(vertexCoords.size());
  for (size_t iF = 0; iF < faces.size(); iF++) {
    for (size_t j = 0; j < 3; j++) {
      size_t vInd = faces[iF][j];
      Vector3 pOrig = vertexCoords[vInd];
      Face origF = origFaceTri[iF];
      Vector3 origBary = baryCoords[iF][j];

      SurfacePoint origP(origF, origBary);
      vertexCoordsOffset[vInd] = bubbleOffset.queryPoint(origP);
    }
  }


  // Store the output here
  std::unique_ptr<PolygonSoupMesh> outSoup(new PolygonSoupMesh({}, vertexCoordsOffset));
  for (size_t iF = 0; iF < faces.size(); iF++) {
    size_t vA = faces[iF][0];
    size_t vB = faces[iF][1];
    size_t vC = faces[iF][2];
    outSoup->polygons.push_back({vA, vB, vC});
  }

  return outSoup;
}


void printMeshQuality(const PolygonSoupMesh& soup, std::ofstream& s) {

  { // Angle things
    std::vector<double> angles;
    std::vector<double> cotans;
    for (size_t iF = 0; iF < soup.polygons.size(); iF++) {

      const std::vector<size_t>& face = soup.polygons[iF];
      size_t D = face.size();

      for (size_t j = 0; j < D; j++) {

        size_t vPrev = face[(j + D - 1) % D];
        size_t vCurr = face[j];
        size_t vNext = face[(j + 1) % D];

        Vector3 pPrev = soup.vertexCoordinates[vPrev];
        Vector3 pCurr = soup.vertexCoordinates[vCurr];
        Vector3 pNext = soup.vertexCoordinates[vNext];

        Vector3 vA = pNext - pCurr;
        Vector3 vB = pPrev - pCurr;

        double oppTan = norm(cross(vA, vB)) / dot(vA, vB);
        double oppAngle = std::fmod(std::atan(oppTan) + 2 * PI, PI);
        angles.push_back(oppAngle);

        double cotan = dot(vA, vB) / norm(cross(vA, vB));
        cotans.push_back(cotan);
      }
    }


    std::sort(angles.begin(), angles.end());
    std::sort(cotans.begin(), cotans.end());

    // cout << tag << "smallestAngleDeg," << (angles.front() * 180 / PI) << endl;
    // cout << tag << "largestAngleDeg," << (angles.back() * 180 / PI) << endl;
    // cout << tag << "smallestCotan," << (cotans.front()) << endl;
    // cout << tag << "largestCotan," << (cotans.back()) << endl;

    s << (angles.front() * 180 / PI) << ",";
    s << (angles.back() * 180 / PI) << ",";
    s << (cotans.front()) << ",";
    s << (cotans.back()) << ",";
  }
}


std::vector<std::array<double, 3>> computeCotanWeights(const PolygonSoupMesh& soup, double denomEps) {

  std::vector<std::array<double, 3>> cotanWs(soup.polygons.size());

  for (size_t iF = 0; iF < soup.polygons.size(); iF++) {
    const std::vector<size_t>& face = soup.polygons[iF];
    if (face.size() != 3) throw std::runtime_error("non-triangular face");

    for (size_t root = 0; root < 3; root++) {

      size_t i = face[root];
      size_t j = face[(root + 1) % 3];
      size_t k = face[(root + 2) % 3];

      Vector3 pI = soup.vertexCoordinates[i];
      Vector3 pJ = soup.vertexCoordinates[j];
      Vector3 pK = soup.vertexCoordinates[k];

      Vector3 vA = pJ - pI;
      Vector3 vB = pK - pI;

      double cotanW = 0.5 * dot(vA, vB) / (norm(cross(vA, vB)) + denomEps);
      cotanWs[iF][root] = cotanW;
    }
  }


  return cotanWs;
}


SparseMatrix<double> buildCotanLaplace(const PolygonSoupMesh& soup,
                                       const std::vector<std::array<double, 3>>& cotanWeights) {

  std::vector<Eigen::Triplet<double>> triplets;

  for (size_t iF = 0; iF < soup.polygons.size(); iF++) {
    const std::vector<size_t>& face = soup.polygons[iF];
    if (face.size() != 3) throw std::runtime_error("non-triangular face");

    for (size_t root = 0; root < 3; root++) {

      size_t i = face[root];
      size_t j = face[(root + 1) % 3];
      size_t k = face[(root + 2) % 3];

      double cotanW = cotanWeights[iF][root];

      triplets.emplace_back(j, k, -cotanW);
      triplets.emplace_back(k, j, -cotanW);
      triplets.emplace_back(j, j, cotanW);
      triplets.emplace_back(k, k, cotanW);
    }
  }


  size_t nV = soup.vertexCoordinates.size();
  SparseMatrix<double> L(nV, nV);
  L.setFromTriplets(triplets.begin(), triplets.end());
  return L;
}

SparseMatrix<double> buildLumpedMass(const PolygonSoupMesh& soup, const std::vector<double>& dualAreas) {
  size_t nV = soup.vertexCoordinates.size();

  std::vector<Eigen::Triplet<double>> triplets;
  for (size_t i = 0; i < nV; i++) {
    triplets.emplace_back(i, i, dualAreas[i]);
  }

  SparseMatrix<double> M(nV, nV);
  M.setFromTriplets(triplets.begin(), triplets.end());
  return M;
}

double computeMeanEdgeLength(const PolygonSoupMesh& soup) {
  // NOTE: More formally this is a mean over halfedge lengths, since we only iterate per-triangle

  double edgeLenSum = 0.;
  size_t edgeLenCount = 0;

  for (size_t iF = 0; iF < soup.polygons.size(); iF++) {
    const std::vector<size_t>& face = soup.polygons[iF];
    if (face.size() != 3) throw std::runtime_error("non-triangular face");

    for (size_t root = 0; root < 3; root++) {

      size_t i = face[root];
      size_t j = face[(root + 1) % 3];

      Vector3 pI = soup.vertexCoordinates[i];
      Vector3 pJ = soup.vertexCoordinates[j];

      double l = norm(pJ - pI);
      edgeLenSum += l;
      edgeLenCount += 1;
    }
  }

  return edgeLenSum / edgeLenCount;
}

std::vector<double> computeVertexDualAreas(const PolygonSoupMesh& soup) {

  std::vector<double> result(soup.vertexCoordinates.size(), 0.);

  for (size_t iF = 0; iF < soup.polygons.size(); iF++) {
    const std::vector<size_t>& face = soup.polygons[iF];
    if (face.size() != 3) throw std::runtime_error("non-triangular face");

    size_t i = face[0];
    size_t j = face[1];
    size_t k = face[2];

    Vector3 pI = soup.vertexCoordinates[i];
    Vector3 pJ = soup.vertexCoordinates[j];
    Vector3 pK = soup.vertexCoordinates[k];

    double area = 0.5 * norm(cross(pJ - pI, pK - pI));

    result[i] += area / 3.;
    result[j] += area / 3.;
    result[k] += area / 3.;
  }

  return result;
}


void mollify(HalfedgeMesh& mesh, EdgeData<double>& edgeLengths, double mollifyFactorRel) {
  // Mean edge length
  double edgeSum = 0.;
  for (Edge e : mesh.edges()) {
    edgeSum += edgeLengths[e];
  }
  double meanEdge = edgeSum / mesh.nEdges();

  double mollifyDelta = meanEdge * mollifyFactorRel;

  // Compute the mollify epsilon
  double mollifyEps = 0.;
  for (Halfedge he : mesh.interiorHalfedges()) {

    double lA = edgeLengths[he.edge()];
    double lB = edgeLengths[he.next().edge()];
    double lC = edgeLengths[he.next().next().edge()];

    double thisEPS = lC - lA - lB + mollifyDelta;
    mollifyEps = std::fmax(mollifyEps, thisEPS);
  }
  double mollifyEpsRel = mollifyEps / meanEdge;

  // Apply the offset
  for (Edge e : mesh.edges()) {
    edgeLengths[e] += mollifyEps;
  }
}
