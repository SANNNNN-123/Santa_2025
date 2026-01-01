//%%writefile a.cpp
// Tree Packer v21 - ENHANCED v19 with SWAP MOVES + MULTI-START
// All n values (1-200) processed in parallel + aggressive exploration
// NEW: Swap move operator, multi-angle restarts, higher temperature SA
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o tree_packer_v21 tree_packer_v21.cpp

#include <bits/stdc++.h>  // Standard C++ library (includes iostream, vector, algorithm, etc.)
#include <omp.h>           // OpenMP for parallel processing
using namespace std;

// ============================================================================
// CONSTANTS
// ============================================================================
constexpr int MAX_N = 200;  // Maximum number of trees to pack (n=1 to 200)
constexpr int NV = 15;      // Number of vertices defining each Christmas tree polygon
constexpr double PI = 3.14159265358979323846; // Pi constant to calculate the degree and radians for rotation  deg * PI / 180.0L

// Tree shape definition: 15 vertices defining the Christmas tree polygon
// These are the base coordinates that get rotated and translated for each tree
// alignas(64) ensures memory alignment for better cache performance
alignas(64) const long double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
alignas(64) const long double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

// ============================================================================
// FAST RANDOM NUMBER GENERATOR
// ============================================================================
// Custom RNG for performance - faster than std::random
// Uses xoshiro128++ algorithm variant
struct FastRNG {
    uint64_t s[2];  // Two 64-bit state variables
    
    // Initialize with seed
    FastRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }
    
    // Rotate left operation
    uint64_t rotl(uint64_t x, int k) {
        // Step 1: Push beads to the left (some fall off)
        uint64_t left_part = (x << k);
        
        // Step 2: Find the beads that would have fallen off 
        // by looking at the other end (64 - k)
        uint64_t right_part = (x >> (64 - k));
        
        // Step 3: Glue them back together with OR (|)
        return left_part | right_part;
    }
    
    // Generate next random number using xoshiro128++ algorithm
    // xoshiro stands for XOR, SHIFT, ROTATE
    // It is a very fast, high-quality PRNG used in simulations
    inline uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1];
        // uint64_t r = s0 + s1; // Original xoshiro128+ calculation
        uint64_t r = rotl(s0 + s1, 17) + s0;  // xoshiro128++ calculation

        s1 ^= s0; // XOR
        
        // original code: 
        // s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); 
        // s[1] = rotl(s1, 37);

        s[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21); // Rotate, XOR, Shift (v++)
        s[1] = rotl(s1, 28); // Rotate (v++)

        return r;
    }
    
    // Random float in [0, 1)
    inline long double rf() { return (next() >> 11) * 0x1.0p-53L; }
    
    // Random float in [-1, 1)
    inline long double rf2() { return rf() * 2.0L - 1.0L; }
    
    // Random integer in [0, n)
    inline int ri(int n) { return next() % n; }
    
    // Gaussian (normal) distribution using Box-Muller transform
    inline long double gaussian() {
        long double u1 = rf() + 1e-10L, u2 = rf();
        return sqrtl(-2.0L * logl(u1)) * cosl(2.0L * PI * u2);
    }
};

// ============================================================================
// POLYGON STRUCTURE
// ============================================================================
// Stores a transformed tree polygon with its bounding box
struct Poly {
    long double px[NV], py[NV];  // Transformed vertex coordinates
    long double x0, y0, x1, y1;  // Bounding box: (x0,y0) bottom-left, (x1,y1) top-right
};

// ============================================================================
// GEOMETRIC TRANSFORMATIONS
// ============================================================================
// Transform tree from base shape to position (cx, cy) with rotation deg degrees
// Applies rotation matrix and translation
inline void getPoly(long double cx, long double cy, long double deg, Poly& q) {
    long double rad = deg * (PI / 180.0L);  // Convert degrees to radians
    long double s = sinl(rad), c = cosl(rad);  // Precompute sin/cos
    
    long double minx = 1e9L, miny = 1e9L, maxx = -1e9L, maxy = -1e9L;
    
    // Transform each vertex: rotation + translation
    for (int i = 0; i < NV; i++) {
        // Rotation matrix: [cos -sin] [x]   Translation: [cx]
        //                   [sin  cos] [y]                [cy]
        long double x = TX[i] * c - TY[i] * s + cx;
        long double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x; q.py[i] = y;
        
        // Track bounding box
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

// ============================================================================
// POINT-IN-POLYGON TEST
// ============================================================================
// Ray casting algorithm: count intersections of horizontal ray from point
// Returns true if point (px, py) is inside polygon q
inline bool pip(long double px, long double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;  // Last vertex
    
    for (int i = 0; i < NV; i++) {
        // Check if ray crosses edge from vertex j to vertex i
        if ((q.py[i] > py) != (q.py[j] > py) &&  // Ray crosses edge in y-direction
            px < (q.px[j] - q.px[i]) * (py - q.py[i]) / (q.py[j] - q.py[i]) + q.px[i])
            in = !in;  // Toggle inside/outside
        j = i;
    }
    return in;
}

// ============================================================================
// LINE SEGMENT INTERSECTION
// ============================================================================
// Check if line segments (ax,ay)-(bx,by) and (cx,cy)-(dx,dy) intersect
// Uses cross product method
inline bool segInt(long double ax, long double ay, long double bx, long double by,
                   long double cx, long double cy, long double dx, long double dy) {
    // Cross products to determine orientation
    long double d1 = (dx-cx)*(ay-cy) - (dy-cy)*(ax-cx);  // Point a relative to segment cd
    long double d2 = (dx-cx)*(by-cy) - (dy-cy)*(bx-cx);  // Point b relative to segment cd
    long double d3 = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);  // Point c relative to segment ab
    long double d4 = (bx-ax)*(dy-ay) - (by-ay)*(dx-ax);  // Point d relative to segment ab
    
    // Segments intersect if points are on opposite sides of each other's lines
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}

// ============================================================================
// OVERLAP DETECTION
// ============================================================================
// Check if two tree polygons overlap
// Uses three methods: bounding box, point-in-polygon, edge intersection
inline bool overlap(const Poly& a, const Poly& b) {
    // Fast rejection: check bounding boxes first
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    
    // Check if any vertex of one polygon is inside the other
    for (int i = 0; i < NV; i++) {
        if (pip(a.px[i], a.py[i], b)) return true;  // Vertex of a inside b
        if (pip(b.px[i], b.py[i], a)) return true;  // Vertex of b inside a
    }
    
    // Check if any edges intersect
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;  // Next vertex (wraps around)
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segInt(a.px[i], a.py[i], a.px[ni], a.py[ni],
                      b.px[j], b.py[j], b.px[nj], b.py[nj])) return true;
        }
    }
    return false;
}

// ============================================================================
// CONFIGURATION STRUCTURE
// ============================================================================
// Stores a complete configuration of n trees
struct Cfg {
    int n;  // Number of trees
    long double x[MAX_N], y[MAX_N], a[MAX_N];  // Position and rotation for each tree
    Poly pl[MAX_N];  // Transformed polygons for each tree
    long double gx0, gy0, gx1, gy1;  // Global bounding box of all trees

    // Update polygon for tree i
    inline void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    
    // Update all trees and global bounding box
    inline void updAll() { for (int i = 0; i < n; i++) upd(i); updGlobal(); }

    // Update global bounding box from all tree bounding boxes
    inline void updGlobal() {
        gx0 = gy0 = 1e9L; gx1 = gy1 = -1e9L;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;  // Leftmost
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;  // Rightmost
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;  // Bottommost
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;  // Topmost
        }
    }

    // Check if tree i overlaps with any other tree
    inline bool hasOvl(int i) const {
        for (int j = 0; j < n; j++)
            if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }

    // Check if any trees overlap
    inline bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return true;
        return false;
    }

    // Calculate side length of square bounding box (max of width/height)
    inline long double side() const { return max(gx1 - gx0, gy1 - gy0); }
    
    // Calculate score: side² / n (lower is better)
    inline long double score() const { long double s = side(); return s * s / n; }

    // Get indices of trees touching the bounding box boundary
    // These are candidates for optimization
    void getBoundary(vector<int>& b) const {
        b.clear();
        long double eps = 0.01L;  // Epsilon for boundary detection
        for (int i = 0; i < n; i++) {
            // Check if tree touches any boundary (left, right, bottom, top)
            if (pl[i].x0 - gx0 < eps || gx1 - pl[i].x1 < eps ||
                pl[i].y0 - gy0 < eps || gy1 - pl[i].y1 < eps)
                b.push_back(i);
        }
    }

    // Remove tree at index, shift others down
    // Used in back propagation optimization
    Cfg removeTree(int removeIdx) const {
        Cfg c;
        c.n = n - 1;
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i != removeIdx) {
                c.x[j] = x[i];
                c.y[j] = y[i];
                c.a[j] = a[i];
                j++;
            }
        }
        c.updAll();
        return c;
    }
};

// ============================================================================
// OPTIMIZATION ALGORITHMS
// ============================================================================

// SQUEEZE: Uniformly scale all trees toward center to reduce bounding box
// Tries progressively smaller scales until overlap occurs
Cfg squeeze(Cfg c) {
    long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;  // Center
    for (long double scale = 0.9995L; scale >= 0.98L; scale -= 0.0005L) {
        Cfg trial = c;
        // Scale each tree position relative to center
        for (int i = 0; i < c.n; i++) {
            trial.x[i] = cx + (c.x[i] - cx) * scale;
            trial.y[i] = cy + (c.y[i] - cy) * scale;
        }
        trial.updAll();
        if (!trial.anyOvl()) c = trial;  // Accept if no overlap
        else break;  // Stop if overlap occurs
    }
    return c;
}

// COMPACTION: Move individual trees toward center in small steps
// Tries multiple step sizes, accepts moves that reduce bounding box
Cfg compaction(Cfg c, int iters) {
    long double bs = c.side();  // Best side length so far
    for (int it = 0; it < iters; it++) {
        long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
        bool improved = false;
        
        for (int i = 0; i < c.n; i++) {
            long double ox = c.x[i], oy = c.y[i];  // Original position
            long double dx = cx - c.x[i], dy = cy - c.y[i];  // Direction to center
            long double d = sqrtl(dx*dx + dy*dy);
            if (d < 1e-6L) continue;  // Skip if already at center
            
            // Try multiple step sizes (larger to smaller)
            for (long double step : {0.02L, 0.008L, 0.003L, 0.001L, 0.0004L}) {
                c.x[i] = ox + dx/d * step; c.y[i] = oy + dy/d * step; c.upd(i);
                if (!c.hasOvl(i)) {
                    c.updGlobal();
                    if (c.side() < bs - 1e-12L) {  // Improvement
                        bs = c.side(); improved = true; ox = c.x[i]; oy = c.y[i];
                    } else {  // Revert
                        c.x[i] = ox; c.y[i] = oy; c.upd(i);
                    }
                } else {  // Overlap, revert
                    c.x[i] = ox; c.y[i] = oy; c.upd(i);
                }
            }
        }
        c.updGlobal();
        if (!improved) break;  // Stop if no improvement
    }
    return c;
}

// LOCAL SEARCH: Fine-tune positions and rotations with small moves
// Tries moving toward center, in 8 directions, and rotations
Cfg localSearch(Cfg c, int maxIter) {
    long double bs = c.side();
    const long double steps[] = {0.01L, 0.004L, 0.0015L, 0.0006L, 0.00025L, 0.0001L};  // Step sizes
    const long double rots[] = {5.0L, 2.0L, 0.8L, 0.3L, 0.1L};  // Rotation angles
    const int dx[] = {1,-1,0,0,1,1,-1,-1};  // 8-directional moves
    const int dy[] = {0,0,1,-1,1,-1,1,-1};

    for (int iter = 0; iter < maxIter; iter++) {
        bool improved = false;
        
        for (int i = 0; i < c.n; i++) {
            long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
            long double ddx = cx - c.x[i], ddy = cy - c.y[i];
            long double dist = sqrtl(ddx*ddx + ddy*ddy);
            
            // Move toward center
            if (dist > 1e-6L) {
                for (long double st : steps) {
                    long double ox = c.x[i], oy = c.y[i];
                    c.x[i] += ddx/dist * st; c.y[i] += ddy/dist * st; c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); }
                    } else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            
            // Try 8-directional moves
            for (long double st : steps) {
                for (int d = 0; d < 8; d++) {
                    long double ox=c.x[i], oy=c.y[i];
                    c.x[i] += dx[d]*st; c.y[i] += dy[d]*st; c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); }
                    } else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            
            // Try rotations
            for (long double rt : rots) {
                for (long double da : {rt, -rt}) {  // Try both directions
                    long double oa = c.a[i]; c.a[i] += da;
                    while (c.a[i] < 0) c.a[i] += 360.0L;
                    while (c.a[i] >= 360.0L) c.a[i] -= 360.0L;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; }
                        else { c.a[i]=oa; c.upd(i); c.updGlobal(); }
                    } else { c.a[i]=oa; c.upd(i); }
                }
            }
        }
        if (!improved) break;
    }
    return c;
}

// V21 NEW: Swap move operator - exchange positions/rotations of two trees
// Can help escape local optima
bool swapTrees(Cfg& c, int i, int j) {
    if (i == j || i >= c.n || j >= c.n) return false;
    swap(c.x[i], c.x[j]);
    swap(c.y[i], c.y[j]);
    swap(c.a[i], c.a[j]);
    c.upd(i);
    c.upd(j);
    return !c.hasOvl(i) && !c.hasOvl(j);  // Valid if no overlaps
}

// ============================================================================
// SIMULATED ANNEALING OPTIMIZATION
// ============================================================================
// Main optimization algorithm using simulated annealing
// Tries 11 different move types, accepts improvements or worse moves with probability
Cfg sa_opt(Cfg c, int iter, long double T0, long double Tm, uint64_t seed) {
    FastRNG rng(seed);
    Cfg best = c, cur = c;  // Best and current configurations
    long double bs = best.side(), cs = bs, T = T0;  // Best/current side, temperature
    long double alpha = powl(Tm / T0, 1.0L / iter);  // Cooling rate
    int noImp = 0;  // No improvement counter

    for (int it = 0; it < iter; it++) {
        int mt = rng.ri(11);  // V21: 11 move types (added swap move)
        long double sc = T / T0;  // Scale factor based on temperature
        
        bool valid = true;

        // Move type 0: Gaussian position perturbation
        if (mt == 0) {
            int i = rng.ri(c.n);
            long double ox = cur.x[i], oy = cur.y[i];
            cur.x[i] += rng.gaussian() * 0.5L * sc;
            cur.y[i] += rng.gaussian() * 0.5L * sc;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        // Move type 1: Move toward center
        else if (mt == 1) {
            int i = rng.ri(c.n);
            long double ox = cur.x[i], oy = cur.y[i];
            long double bcx = (cur.gx0+cur.gx1)/2.0L, bcy = (cur.gy0+cur.gy1)/2.0L;
            long double dx = bcx - cur.x[i], dy = bcy - cur.y[i];
            long double d = sqrtl(dx*dx + dy*dy);
            if (d > 1e-6L) { cur.x[i] += dx/d * rng.rf() * 0.6L * sc; cur.y[i] += dy/d * rng.rf() * 0.6L * sc; }
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        // Move type 2: Rotation adjustment
        else if (mt == 2) {
            int i = rng.ri(c.n);
            long double oa = cur.a[i];
            cur.a[i] += rng.gaussian() * 80.0L * sc;
            while (cur.a[i] < 0) cur.a[i] += 360.0L;
            while (cur.a[i] >= 360.0L) cur.a[i] -= 360.0L;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        // Move type 3: Combined position + rotation
        else if (mt == 3) {
            int i = rng.ri(c.n);
            long double ox=cur.x[i], oy=cur.y[i], oa=cur.a[i];
            cur.x[i] += rng.rf2() * 0.5L * sc;
            cur.y[i] += rng.rf2() * 0.5L * sc;
            cur.a[i] += rng.rf2() * 60.0L * sc;
            while (cur.a[i] < 0) cur.a[i] += 360.0L;
            while (cur.a[i] >= 360.0L) cur.a[i] -= 360.0L;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        // Move type 4: Optimize boundary trees (trees touching bounding box)
        else if (mt == 4) {
            vector<int> boundary; cur.getBoundary(boundary);
            if (!boundary.empty()) {
                int i = boundary[rng.ri(boundary.size())];
                long double ox=cur.x[i], oy=cur.y[i], oa=cur.a[i];
                long double bcx = (cur.gx0+cur.gx1)/2.0L, bcy = (cur.gy0+cur.gy1)/2.0L;
                long double dx = bcx - cur.x[i], dy = bcy - cur.y[i];
                long double d = sqrtl(dx*dx + dy*dy);
                if (d > 1e-6L) { cur.x[i] += dx/d * rng.rf() * 0.7L * sc; cur.y[i] += dy/d * rng.rf() * 0.7L * sc; }
                cur.a[i] += rng.rf2() * 50.0L * sc;
                while (cur.a[i] < 0) cur.a[i] += 360.0L;
                while (cur.a[i] >= 360.0L) cur.a[i] -= 360.0L;
                cur.upd(i);
                if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i); valid=false; }
            } else valid = false;
        }
        // Move type 5: Global scaling (squeeze)
        else if (mt == 5) {
            long double factor = 1.0L - rng.rf() * 0.004L * sc;
            long double cx = (cur.gx0 + cur.gx1) / 2.0L, cy = (cur.gy0 + cur.gy1) / 2.0L;
            Cfg trial = cur;
            for (int i = 0; i < c.n; i++) { trial.x[i] = cx + (cur.x[i] - cx) * factor; trial.y[i] = cy + (cur.y[i] - cy) * factor; }
            trial.updAll();
            if (!trial.anyOvl()) cur = trial; else valid = false;
        }
        // Move type 6: Lévy flight (long jumps for exploration)
        else if (mt == 6) {
            int i = rng.ri(c.n);
            long double ox=cur.x[i], oy=cur.y[i];
            long double levy = powl(rng.rf() + 0.001L, -1.3L) * 0.008L;  // Power-law distribution
            cur.x[i] += rng.rf2() * levy; cur.y[i] += rng.rf2() * levy; cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        // Move type 7: Coupled move (move two adjacent trees together)
        else if (mt == 7 && c.n > 1) {
            int i = rng.ri(c.n), j = (i + 1) % c.n;
            long double oxi=cur.x[i], oyi=cur.y[i], oxj=cur.x[j], oyj=cur.y[j];
            long double dx = rng.rf2() * 0.3L * sc, dy = rng.rf2() * 0.3L * sc;
            cur.x[i]+=dx; cur.y[i]+=dy; cur.x[j]+=dx; cur.y[j]+=dy;
            cur.upd(i); cur.upd(j);
            if (cur.hasOvl(i) || cur.hasOvl(j)) { cur.x[i]=oxi; cur.y[i]=oyi; cur.x[j]=oxj; cur.y[j]=oyj; cur.upd(i); cur.upd(j); valid=false; }
        }
        // Move type 10: V21 NEW - Swap move
        else if (mt == 10 && c.n > 1) {
            int i = rng.ri(c.n), j = rng.ri(c.n);
            Cfg old = cur;
            if (!swapTrees(cur, i, j)) {
                cur = old;
                valid = false;
            }
        }
        // Move type 8,9: Small random perturbation
        else {
            int i = rng.ri(c.n);
            long double ox=cur.x[i], oy=cur.y[i];
            cur.x[i] += rng.rf2() * 0.002L; cur.y[i] += rng.rf2() * 0.002L; cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }

        // If move invalid, cool down and continue
        if (!valid) { noImp++; T *= alpha; if (T < Tm) T = Tm; continue; }

        // Evaluate new configuration
        cur.updGlobal();
        long double ns = cur.side();
        long double delta = ns - cs;  // Change in side length

        // Simulated annealing acceptance: accept if better, or with probability if worse
        if (delta < 0 || rng.rf() < expl(-delta / T)) {  // expl = exp (typo in original)
            cs = ns;
            if (ns < bs) { bs = ns; best = cur; noImp = 0; }  // New best
            else noImp++;
        } else { cur = best; cs = bs; noImp++; }  // Reject, revert to best

        // Reheat if stuck (no improvement for 200 iterations)
        if (noImp > 200) { T = min(T * 5.0L, T0); noImp = 0; }
        
        // Cool down
        T *= alpha;
        if (T < Tm) T = Tm;  // Minimum temperature
    }
    return best;
}

// ============================================================================
// PERTURBATION
// ============================================================================
// Create a perturbed starting configuration for multi-start optimization
// Randomly moves some trees, then fixes any overlaps
Cfg perturb(Cfg c, long double str, FastRNG& rng) {
    Cfg original = c;
    int np = max(1, (int)(c.n * 0.08L + str * 3.0L));  // Number of trees to perturb
    
    // Perturb random trees
    for (int k = 0; k < np; k++) {
        int i = rng.ri(c.n);
        c.x[i] += rng.gaussian() * str * 0.5L;
        c.y[i] += rng.gaussian() * str * 0.5L;
        c.a[i] += rng.gaussian() * 30.0L;
        while (c.a[i] < 0) c.a[i] += 360.0L;
        while (c.a[i] >= 360.0L) c.a[i] -= 360.0L;
    }
    c.updAll();
    
    // Fix overlaps by moving overlapping trees away from center
    for (int iter = 0; iter < 150; iter++) {
        bool fixed = true;
        for (int i = 0; i < c.n; i++) {
            if (c.hasOvl(i)) {
                fixed = false;
                long double cx = (c.gx0+c.gx1)/2.0L, cy = (c.gy0+c.gy1)/2.0L;
                long double dx = c.x[i] - cx, dy = c.y[i] - cy;
                long double d = sqrtl(dx*dx + dy*dy);
                if (d > 1e-6L) { c.x[i] += dx/d*0.02L; c.y[i] += dy/d*0.02L; }
                c.a[i] += rng.rf2() * 15.0L;
                while (c.a[i] < 0) c.a[i] += 360.0L;
                while (c.a[i] >= 360.0L) c.a[i] -= 360.0L;
                c.upd(i);
            }
        }
        if (fixed) break;
    }
    c.updGlobal();
    if (c.anyOvl()) return original;  // Return original if still has overlaps
    return c;
}

// ============================================================================
// PARALLEL OPTIMIZATION
// ============================================================================
// Run multiple restarts in parallel, each with different starting point
// Uses OpenMP for parallelization
Cfg optimizeParallel(Cfg c, int iters, int restarts) {
    Cfg globalBest = c;
    long double globalBestSide = c.side();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();  // Thread ID
        FastRNG rng(42 + tid * 1000 + c.n);  // Unique seed per thread
        Cfg localBest = c;
        long double localBestSide = c.side();

        #pragma omp for schedule(dynamic)  // Dynamic scheduling for load balancing
        for (int r = 0; r < restarts; r++) {
            Cfg start;
            if (r == 0) {
                start = c;  // First restart uses original
            }
            // V21: Every 4th restart, try rotating all trees by a fixed angle
            // This explores different orientations systematically
            else if (r % 4 == 0 && r < restarts / 2) {
                start = c;
                long double angleOffset = (r / 4) * 45.0L;  // Try 0, 45, 90, 135, etc.
                for (int i = 0; i < start.n; i++) {
                    start.a[i] += angleOffset;
                    while (start.a[i] >= 360.0L) start.a[i] -= 360.0L;
                }
                start.updAll();
                if (start.anyOvl()) {
                    start = perturb(c, 0.02L + 0.02L * (r % 8), rng);
                    if (start.anyOvl()) continue;
                }
            }
            else {
                start = perturb(c, 0.02L + 0.02L * (r % 8), rng);
                if (start.anyOvl()) continue;
            }

            uint64_t seed = 42 + r * 1000 + tid * 100000 + c.n;
            Cfg o = sa_opt(start, iters, 3.0L, 0.0000005L, seed);  // V21: Increased T0 from 2.5 to 3.0
            o = squeeze(o);      // Post-process: squeeze
            o = compaction(o, 50);  // Post-process: compaction
            o = localSearch(o, 80);  // Post-process: local search

            if (!o.anyOvl() && o.side() < localBestSide) {
                localBestSide = o.side();
                localBest = o;
            }
        }

        // Thread-safe update of global best
        #pragma omp critical
        {
            if (!localBest.anyOvl() && localBestSide < globalBestSide) {
                globalBestSide = localBestSide;
                globalBest = localBest;
            }
        }
    }

    // Final refinement
    globalBest = squeeze(globalBest);
    globalBest = compaction(globalBest, 80);
    globalBest = localSearch(globalBest, 150);

    if (globalBest.anyOvl()) return c;  // Return original if invalid
    return globalBest;
}

// ============================================================================
// FILE I/O
// ============================================================================

// Load configurations from CSV file
map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) return cfg;
    string ln; getline(f, ln);  // Skip header
    map<int, vector<tuple<int,long double,long double,long double>>> data;
    
    while (getline(f, ln)) {
        // Parse CSV: id,x,y,deg
        size_t p1=ln.find(','), p2=ln.find(',',p1+1), p3=ln.find(',',p2+1);
        string id=ln.substr(0,p1), xs=ln.substr(p1+1,p2-p1-1), ys=ln.substr(p2+1,p3-p2-1), ds=ln.substr(p3+1);
        
        // Remove 's' prefix if present
        if(!xs.empty() && xs[0]=='s') xs=xs.substr(1);
        if(!ys.empty() && ys[0]=='s') ys=ys.substr(1);
        if(!ds.empty() && ds[0]=='s') ds=ds.substr(1);
        
        int n=stoi(id.substr(0,3)), idx=stoi(id.substr(4));  // Parse "001_0" -> n=1, idx=0
        data[n].push_back({idx, stold(xs), stold(ys), stold(ds)});
    }
    
    // Build configurations
    for (auto& [n,v] : data) {
        Cfg c; c.n = n;
        for (auto& [i,x,y,d] : v) if (i < n) { c.x[i]=x; c.y[i]=y; c.a[i]=d; }
        c.updAll();
        cfg[n] = c;
    }
    return cfg;
}

// Save configurations to CSV file
void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(17) << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        if (cfg.count(n)) {
            const Cfg& c = cfg.at(n);
            for (int i = 0; i < n; i++)
                f << setfill('0') << setw(3) << n << "_" << i << ",s" << c.x[i] << ",s" << c.y[i] << ",s" << c.a[i] << "\n";
        }
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main(int argc, char** argv) {
    // Default parameters
    string in="submission.csv", out="submission_v21.csv";
    int iters=15000, restarts=16;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a=="-i" && i+1<argc) in=argv[++i];
        else if (a=="-o" && i+1<argc) out=argv[++i];
        else if (a=="-n" && i+1<argc) iters=stoi(argv[++i]);
        else if (a=="-r" && i+1<argc) restarts=stoi(argv[++i]);
    }

    int numThreads = omp_get_max_threads();
    printf("Tree Packer v21 - ENHANCED (%d threads)\n", numThreads);
    printf("NEW: Swap moves, multi-angle restarts, higher SA temperature\n");
    printf("Iterations: %d, Restarts: %d\n", iters, restarts);
    printf("Processing all n=1..200 concurrently\n");
    printf("Loading %s...\n", in.c_str());

    auto cfg = loadCSV(in);
    if (cfg.empty()) { printf("No data!\n"); return 1; }
    printf("Loaded %d configs\n", (int)cfg.size());

    // Calculate initial total score
    long double init = 0;
    for (auto& [n,c] : cfg) init += c.score();
    printf("Initial: %.6Lf\n\nPhase 1: Parallel optimization...\n\n", init);

    auto t0 = chrono::high_resolution_clock::now();
    map<int, Cfg> res;
    int totalImproved = 0;

    // ========================================================================
    // PHASE 1: Main optimization - PARALLEL OVER ALL N
    // ========================================================================
    vector<int> nvals;
    for (auto& [n,c] : cfg) nvals.push_back(n);

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < (int)nvals.size(); idx++) {
        int n = nvals[idx];
        Cfg c = cfg[n];
        long double os = c.score();

        // Adaptive parameters: smaller n gets more iterations
        int it = iters, r = restarts;
        if (n <= 10) { it = (int)(iters * 2.5); r = restarts * 2; }
        else if (n <= 30) { it = (int)(iters * 1.8); r = (int)(restarts * 1.5); }
        else if (n <= 60) { it = (int)(iters * 1.3); r = restarts; }
        else if (n > 150) { it = (int)(iters * 0.7); r = (int)(restarts * 0.8); }

        Cfg o = optimizeParallel(c, it, max(4, r));

        // Smart overlap handling: prefer non-overlapping configs
        bool o_ovl = o.anyOvl();
        bool c_ovl = c.anyOvl();

        if (!c_ovl && o_ovl) {
            // Original is valid but optimized has overlap, use original
            o = c;
        } else if (c_ovl && !o_ovl) {
            // Original has overlap but optimized doesn't, use optimized even if worse
            // Keep o (no change needed)
        } else if (!c_ovl && !o_ovl && o.side() > c.side() + 1e-14L) {
            // Both valid, but optimized is worse, use original
            o = c;
        } else if (c_ovl && o_ovl) {
            // Both have overlap, use the one with smaller side
            if (o.side() > c.side() + 1e-14L) {
                o = c;
            }
        }

        long double ns = o.score();

        #pragma omp critical
        {
            res[n] = o;
            if (c_ovl && !o_ovl) {
                printf("n=%3d: %.6Lf -> %.6Lf (FIXED OVERLAP, %.4Lf%%)\n", n, os, ns, (os-ns)/os*100.0L);
                fflush(stdout);
                totalImproved++;
            } else if (o_ovl) {
                printf("n=%3d: WARNING - still has overlap! (score %.6Lf)\n", n, ns);
                fflush(stdout);
            } else if (ns < os - 1e-10L) {
                printf("n=%3d: %.6Lf -> %.6Lf (%.4Lf%%)\n", n, os, ns, (os-ns)/os*100.0L);
                fflush(stdout);
                totalImproved++;
            }
        }
    }

    // ========================================================================
    // PHASE 2: AGGRESSIVE BACK PROPAGATION
    // ========================================================================
    // If side(k) < side(k-1), try removing trees from k-config to improve (k-1)
    // This exploits the fact that sometimes fewer trees can fit better
    printf("\nPhase 2: Aggressive back propagation (removing trees)...\n\n");

    int backPropImproved = 0;
    bool changed = true;
    int passNum = 0;

    while (changed && passNum < 10) {
        changed = false;
        passNum++;

        // Check adjacent pairs: if k trees fit better than (k-1), try removing from k
        for (int k = 200; k >= 2; k--) {
            if (!res.count(k) || !res.count(k-1)) continue;

            long double sideK = res[k].side();
            long double sideK1 = res[k-1].side();

            // If k trees fit in smaller box than (k-1) trees
            if (sideK < sideK1 - 1e-12L) {
                // Try removing each tree from k-config
                Cfg& cfgK = res[k];
                long double bestSide = sideK1;
                Cfg bestCfg = res[k-1];

                #pragma omp parallel
                {
                    long double localBestSide = bestSide;
                    Cfg localBestCfg = bestCfg;

                    #pragma omp for schedule(dynamic)
                    for (int removeIdx = 0; removeIdx < k; removeIdx++) {
                        Cfg reduced = cfgK.removeTree(removeIdx);

                        if (!reduced.anyOvl()) {
                            // Optimize the reduced config
                            reduced = squeeze(reduced);
                            reduced = compaction(reduced, 60);
                            reduced = localSearch(reduced, 100);

                            if (!reduced.anyOvl() && reduced.side() < localBestSide) {
                                localBestSide = reduced.side();
                                localBestCfg = reduced;
                            }
                        }
                    }

                    #pragma omp critical
                    {
                        if (localBestSide < bestSide) {
                            bestSide = localBestSide;
                            bestCfg = localBestCfg;
                        }
                    }
                }

                if (bestSide < sideK1 - 1e-12L && !bestCfg.anyOvl()) {
                    long double oldScore = res[k-1].score();
                    long double newScore = bestCfg.score();
                    res[k-1] = bestCfg;
                    printf("n=%3d: %.6Lf -> %.6Lf (from n=%d removal, %.4Lf%%)\n",
                           k-1, oldScore, newScore, k, (oldScore-newScore)/oldScore*100.0L);
                    fflush(stdout);
                    backPropImproved++;
                    changed = true;
                }
            }
        }

        // Also check k+2, k+3 etc for potential improvements
        for (int k = 200; k >= 3; k--) {
            for (int src = k + 1; src <= min(200, k + 5); src++) {
                if (!res.count(src) || !res.count(k)) continue;

                long double sideSrc = res[src].side();
                long double sideK = res[k].side();

                if (sideSrc < sideK - 1e-12L) {
                    // Try removing (src-k) trees from src-config
                    int toRemove = src - k;
                    Cfg cfgSrc = res[src];

                    // Generate combinations to try (sample if too many)
                    vector<vector<int>> combos;
                    if (toRemove == 1) {
                        for (int i = 0; i < src; i++) combos.push_back({i});
                    } else if (toRemove == 2 && src <= 50) {
                        for (int i = 0; i < src; i++)
                            for (int j = i+1; j < src; j++)
                                combos.push_back({i, j});
                    } else {
                        // Random sampling for larger combinations
                        FastRNG rng(k * 1000 + src);
                        for (int t = 0; t < min(200, src * 3); t++) {
                            vector<int> combo;
                            set<int> used;
                            for (int r = 0; r < toRemove; r++) {
                                int idx;
                                do { idx = rng.ri(src); } while (used.count(idx));
                                used.insert(idx);
                                combo.push_back(idx);
                            }
                            sort(combo.begin(), combo.end());
                            combos.push_back(combo);
                        }
                    }

                    long double bestSide = sideK;
                    Cfg bestCfg = res[k];

                    #pragma omp parallel
                    {
                        long double localBestSide = bestSide;
                        Cfg localBestCfg = bestCfg;

                        #pragma omp for schedule(dynamic)
                        for (int ci = 0; ci < (int)combos.size(); ci++) {
                            Cfg reduced = cfgSrc;

                            // Remove trees in reverse order to maintain indices
                            vector<int> toRem = combos[ci];
                            sort(toRem.rbegin(), toRem.rend());
                            for (int idx : toRem) {
                                reduced = reduced.removeTree(idx);
                            }

                            if (!reduced.anyOvl()) {
                                reduced = squeeze(reduced);
                                reduced = compaction(reduced, 50);
                                reduced = localSearch(reduced, 80);

                                if (!reduced.anyOvl() && reduced.side() < localBestSide) {
                                    localBestSide = reduced.side();
                                    localBestCfg = reduced;
                                }
                            }
                        }

                        #pragma omp critical
                        {
                            if (localBestSide < bestSide) {
                                bestSide = localBestSide;
                                bestCfg = localBestCfg;
                            }
                        }
                    }

                    if (bestSide < sideK - 1e-12L && !bestCfg.anyOvl()) {
                        long double oldScore = res[k].score();
                        long double newScore = bestCfg.score();
                        res[k] = bestCfg;
                        printf("n=%3d: %.6Lf -> %.6Lf (from n=%d removal, %.4Lf%%)\n",
                               k, oldScore, newScore, src, (oldScore-newScore)/oldScore*100.0L);
                        fflush(stdout);
                        backPropImproved++;
                        changed = true;
                    }
                }
            }
        }

        if (changed) printf("Pass %d complete, continuing...\n", passNum);
    }

    // ========================================================================
    // RESULTS
    // ========================================================================
    auto t1 = chrono::high_resolution_clock::now();
    long double el = chrono::duration_cast<chrono::milliseconds>(t1-t0).count() / 1000.0L;

    long double fin = 0;
    for (auto& [n,c] : res) fin += c.score();

    printf("\n========================================\n");
    printf("Initial: %.6Lf\nFinal:   %.6Lf\n", init, fin);
    printf("Improve: %.6Lf (%.4Lf%%)\n", init-fin, (init-fin)/init*100.0L);
    printf("Phase 1 improved: %d configs\n", totalImproved);
    printf("Phase 2 back-prop improved: %d configs\n", backPropImproved);
    printf("Time:    %.1Lfs (with %d threads)\n", el, numThreads);
    printf("========================================\n");

    saveCSV(out, res);
    printf("Saved %s\n", out.c_str());
    return 0;
}
