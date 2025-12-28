//%%writefile bp.cpp
// Backward Propagation Optimizer
// Based on: https://www.kaggle.com/code/guntasdhanjal/santa-2025-simple-optimization-v2
//
// Key idea: If removing one tree from N-tree config gives better (N-1)-tree config, propagate it backward
// Compile: g++ -O3 -std=c++17 -o bp bp.cpp

#include <bits/stdc++.h>  // Standard C++ library (includes iostream, vector, algorithm, etc.)
using namespace std;

// ============================================================================
// CONSTANTS
// ============================================================================
constexpr int MAX_N = 200;  // Maximum number of trees to pack (n=1 to 200)
constexpr int NV = 15;      // Number of vertices defining each Christmas tree polygon
constexpr double PI = 3.14159265358979323846;

// Tree shape definition: 15 vertices defining the Christmas tree polygon
// These are the base coordinates that get rotated and translated for each tree
// alignas(64) ensures memory alignment for better cache performance
alignas(64) const long double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
alignas(64) const long double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

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

    // Calculate global bounding box from all tree bounding boxes
    inline void calc_bounds() {
        gx0 = gy0 = 1e9L;  // Initialize to large values
        gx1 = gy1 = -1e9L;  // Initialize to small values
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;  // Leftmost
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;  // Bottommost
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;  // Rightmost
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;  // Topmost
        }
    }

    // Calculate side length of square bounding box (max of width/height)
    inline long double side() const {
        return max(gx1 - gx0, gy1 - gy0);
    }

    // Remove tree at index idx by shifting remaining trees down
    // Used in backward propagation to test removing trees
    void remove_tree(int idx) {
        // Shift all trees after idx one position left
        for (int i = idx; i < n - 1; i++) {
            x[i] = x[i + 1];
            y[i] = y[i + 1];
            a[i] = a[i + 1];
            pl[i] = pl[i + 1];
        }
        n--;  // Decrease tree count
    }

    // Rebuild all polygons and recalculate bounding box
    // Called after modifying tree positions/rotations
    void rebuild_polys() {
        for (int i = 0; i < n; i++) {
            upd(i);  // Update each tree's polygon
        }
        calc_bounds();  // Recalculate global bounding box
    }
};

// ============================================================================
// GLOBAL STORAGE
// ============================================================================
// Global storage for all configurations
// configs[n] stores the best n-tree configuration
Cfg configs[MAX_N + 1];
long double best_sides[MAX_N + 1];  // Cache of best side length for each n

// ============================================================================
// FILE I/O
// ============================================================================

// Load configurations from CSV file
// Format: id,x,y,deg (e.g., "001_0,s0.5,s0.3,s45.0")
void parse_csv(const string& filename) {
    ifstream f(filename);
    string line;
    getline(f, line);  // Skip header line

    // Temporary storage: map from n to list of (index, x, y, angle) tuples
    map<int, vector<tuple<long double, long double, long double>>> data;

    // Parse each line
    while (getline(f, line)) {
        stringstream ss(line);
        string id_str, x_str, y_str, deg_str;

        // Split CSV line by commas
        getline(ss, id_str, ',');
        getline(ss, x_str, ',');
        getline(ss, y_str, ',');
        getline(ss, deg_str);

        // Parse id like "010_0" -> n=10, index=0
        int n = stoi(id_str.substr(0, 3));

        // Remove 's' prefix from coordinates (Kaggle format)
        long double x = stold(x_str.substr(1));
        long double y = stold(y_str.substr(1));
        long double deg = stold(deg_str.substr(1));

        // Store in temporary map
        data[n].push_back({x, y, deg});
    }

    // Populate configs array from parsed data
    for (auto& [n, trees] : data) {
        configs[n].n = n;
        for (int i = 0; i < n; i++) {
            auto [x, y, a] = trees[i];
            configs[n].x[i] = x;
            configs[n].y[i] = y;
            configs[n].a[i] = a;
        }
        // Rebuild polygons and calculate initial bounding box
        configs[n].rebuild_polys();
        best_sides[n] = configs[n].side();  // Cache initial side length
    }
}

// Save configurations to CSV file
// Format: id,x,y,deg (e.g., "001_0,s0.5,s0.3,s45.0")
void save_csv(const string& filename) {
    ofstream f(filename);
    f << "id,x,y,deg\n";  // Header
    f << fixed << setprecision(17);  // High precision for coordinates

    // Write all configurations (n=1 to 200)
    for (int n = 1; n <= MAX_N; n++) {
        for (int i = 0; i < configs[n].n; i++) {
            // Format: "001_0" with zero-padding
            f << setw(3) << setfill('0') << n << "_" << i << ",";
            // Add 's' prefix (Kaggle format)
            f << "s" << configs[n].x[i] << ",";
            f << "s" << configs[n].y[i] << ",";
            f << "s" << configs[n].a[i] << "\n";
        }
    }
}

// ============================================================================
// SCORE CALCULATION
// ============================================================================
// Calculate total score: sum of (side² / n) for all n=1 to 200
// Lower score is better
long double calc_total_score() {
    long double score = 0.0L;
    for (int n = 1; n <= MAX_N; n++) {
        long double side = configs[n].side();
        score += (side * side) / n;  // Score formula: side² / n
    }
    return score;
}

// ============================================================================
// BOUNDARY DETECTION
// ============================================================================
// Find indices of trees that touch the bounding box boundary
// These trees are candidates for removal since they define the box size
vector<int> get_bbox_touching_tree_indices(const Cfg& cfg) {
    vector<int> touching_indices;
    const long double eps = 1e-9L;  // Epsilon for floating-point comparison
    
    for (int i = 0; i < cfg.n; i++) {
        const Poly& p = cfg.pl[i];
        bool touches = false;
        
        // Check if tree touches left boundary: tree's left edge aligns with global left
        // Also verify tree overlaps with bounding box in y-direction
        if (abs(p.x0 - cfg.gx0) < eps && p.y1 >= cfg.gy0 - eps && p.y0 <= cfg.gy1 + eps) {
            touches = true;
        }
        // Check if tree touches right boundary: tree's right edge aligns with global right
        if (abs(p.x1 - cfg.gx1) < eps && p.y1 >= cfg.gy0 - eps && p.y0 <= cfg.gy1 + eps) {
            touches = true;
        }
        // Check if tree touches bottom boundary: tree's bottom edge aligns with global bottom
        if (abs(p.y0 - cfg.gy0) < eps && p.x1 >= cfg.gx0 - eps && p.x0 <= cfg.gx1 + eps) {
            touches = true;
        }
        // Check if tree touches top boundary: tree's top edge aligns with global top
        if (abs(p.y1 - cfg.gy1) < eps && p.x1 >= cfg.gx0 - eps && p.x0 <= cfg.gx1 + eps) {
            touches = true;
        }
        
        if (touches) {
            touching_indices.push_back(i);
        }
    }
    
    // If no trees touch boundary (shouldn't happen in practice), return all indices as fallback
    if (touching_indices.empty()) {
        for (int i = 0; i < cfg.n; i++) {
            touching_indices.push_back(i);
        }
    }
    
    return touching_indices;
}

// ============================================================================
// BACKWARD PROPAGATION ALGORITHM
// ============================================================================
// Main optimization algorithm: propagate improvements from larger to smaller configs
// 
// Strategy:
// 1. Start from n=200 and work backward to n=2
// 2. For each n, try removing trees from the n-tree config
// 3. If removing a tree gives a better (n-1)-tree config, save it
// 4. Continue removing trees until no more improvements
//
// Why this works:
// - Sometimes a well-optimized n-tree config can yield a better (n-1)-tree config
// - Boundary trees are good candidates for removal (they define box size)
// - This fixes inconsistencies where smaller configs are worse than larger ones
void backward_propagation() {
    cout << "Starting Backward Propagation...\n";
    cout << fixed << setprecision(8) << "Initial score: " << calc_total_score() << "\n\n";

    int total_improvements = 0;

    // Go from N=200 down to N=2 (backward direction)
    for (int n = MAX_N; n >= 2; n--) {

        // Start with a working copy of the n-tree configuration
        Cfg candidate = configs[n];

        // Keep removing trees until we can't improve anymore
        while (candidate.n > 1) {
            int target_size = candidate.n - 1;  // Size after removing one tree
            long double best_current_side = best_sides[target_size];  // Current best for target_size
            long double best_new_side = 1e9L;  // Best side found by removing a tree
            int best_tree_to_delete = -1;  // Index of best tree to remove

            // Get trees that touch the bounding box boundary
            // These are good candidates for removal since they define the box size
            vector<int> touching_indices = get_bbox_touching_tree_indices(candidate);

            // Try deleting each boundary-touching tree
            for (int tree_idx : touching_indices) {
                // Create a test copy
                Cfg test_candidate = candidate;
                test_candidate.remove_tree(tree_idx);  // Remove tree
                test_candidate.calc_bounds();  // Recalculate bounding box

                long double test_side = test_candidate.side();

                // Track the best deletion (smallest side length)
                if (test_side < best_new_side) {
                    best_new_side = test_side;
                    best_tree_to_delete = tree_idx;
                }
            }

            // If we found a deletion candidate, always remove it and continue
            if (best_tree_to_delete != -1) {
                // Remove the best tree from candidate
                candidate.remove_tree(best_tree_to_delete);
                candidate.calc_bounds();

                // If this improves the target_size configuration, save it
                if (best_new_side < best_current_side) {
                    cout << "improved " << candidate.n << " from n=" << n << " " << best_current_side << " -> " << best_new_side << "\n";
                    configs[target_size] = candidate;  // Save improved config
                    best_sides[target_size] = best_new_side;  // Update cached side length
                    total_improvements++;
                }
                // Continue the loop even if not better than stored - keep optimizing
                // This allows us to continue removing trees and potentially find better configs
            } else {
                // Can't find any valid deletion, stop for this configuration
                break;
            }
        }
    }

    // Calculate and report final results
    long double final_score = calc_total_score();
    cout << "\n\nBackward Propagation Complete!\n";
    cout << "Total improvements: " << total_improvements << "\n";
    cout << fixed << setprecision(12) << "Final score: " << final_score << "\n";
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main(int argc, char** argv) {
    // Check command line arguments
    if (argc < 3) {
        cout << "Usage: ./bp input.csv output.csv\n";
        return 1;
    }

    string input_file = argv[1];   // Input CSV file with tree configurations
    string output_file = argv[2];  // Output CSV file for optimized configurations

    cout << "Backward Propagation Optimizer\n";
    cout << "===============================\n";
    cout << "Loading " << input_file << "...\n";

    // Load configurations from input file
    parse_csv(input_file);

    cout << "Loaded " << MAX_N << " configurations\n";

    // Run backward propagation optimization
    backward_propagation();

    // Save optimized configurations to output file
    cout << "Saving to " << output_file << "...\n";
    save_csv(output_file);

    cout << "Done!\n";

    return 0;
}
