# FastRNG Initialization Explained

This document explains how the `FastRNG` struct initializes its internal state (`s[0]` and `s[1]`) from a single seed value.

## The Problem: Why not just use normal random?
In C++, the standard `rand()` function is like a **Toyota Corolla**.
*   It's reliable and easy to use.
*   **But it's slow.** It does a lot of safety checks and math that takes time.
*   **It's weak.** The quality of randomness isn't great for scientific simulations. (Patterns can repeat).

We are building a **Formula 1 Simulation**.
We need to generate **100 Million random numbers per second** to pack these trees. If we used the Corolla (`rand()`), the simulation would take 3 days. With this custom engine (`FastRNG`), it takes 10 minutes.

## The Concept
We need to turn one simple number (the `seed`, e.g., 42) into two completely different, chaotic 64-bit numbers to start the Random Number Generator. To do this, we use "Magic Constants" (derived from SplitMix64) to scramble the seed in two different ways.

## Visual Flowchart

```mermaid
flowchart LR
    Seed((SEED: 42))
    
    subgraph "The Magic Constants"
    MagicK1["Magic Lighter Fluid #1\n(0x853c...)"]
    MagicK2["Magic Lighter Fluid #2\n(0x9e37... & 0xc4ce...)"]
    end

    subgraph "State Memory (s[2])"
    s0["s[0]"]
    s1["s[1]"]
    end

    Seed -- "XOR (^)" --> MagicK1
    MagicK1 --> s0

    Seed -- "MULTIPLY (*)" --> MagicK2
    MagicK2 -- "XOR (^)" --> s1

    style Seed fill:#f9f,stroke:#333
    style s0 fill:#bfc,stroke:#333
    style s1 fill:#bfc,stroke:#333
    style MagicK1 fill:#eee,stroke:#333
    style MagicK2 fill:#eee,stroke:#333
```

## The Code Mapping

| Component | Code | Description |
| :--- | :--- | :--- |
| **State 0** | `s[0] = seed ^ 0x853...` | Simple XOR mix to get the first chunk. |
| **State 1** | `s[1] = (seed * 0x9e3...) ^ 0xc4c...` | Multiplicative mix to get a divergent second chunk. |


```cpp
// The "Long Written" Version
uint64_t rotl(uint64_t x, int k) {
    // Step 1: Push beads to the left (some fall off)
    uint64_t left_part = (x << k);
    
    // Step 2: Find the beads that would have fallen off 
    // by looking at the other end (64 - k)
    uint64_t right_part = (x >> (64 - k));
    
    // Step 3: Glue them back together with OR (|)
    return left_part | right_part;
}
```


### Case Study: Cutting 4 Cards (`k=4`)
You want to move the **Top 4** cards to the **Bottom**.

To do this, we perform three steps:

1.  **Left Shift (`x << 4`)**: Take the **Original Deck**. Shove it Left. Keep the **Body** (Bottom 60). The **Top 4** are trashed.
2.  **Right Shift (`x >> 60`)**: Take the **Original Deck**. Shove it Right. Keep the **Top 4**. The **Body** (Bottom 60) is trashed.
**3. The Merge (`|`)**:
*   **Combine:** [Bottom 60 Cards] + [Top 4 Cards]
*   **Result:** The Top 4 are now at the bottom. The deck is rotated!

### Conclusion
*   **Left Shift (`<<`)**: Trashes the **Top 4**.
*   **Right Shift (`>>`)**: Trashes the **Bottom 60**.



We used a precise combination of Spins (rotl), Flips (XOR), and Shoves (<<) to ensure that s[0] and s[1] change unpredictably every single time you call the function.

## The Helper Functions
These convenience tools use the raw output from `next()` to give us useful numbers for the simulation.

| Function | Code | Goal | How it works |
| :--- | :--- | :--- | :--- |
| **`rf()`** | `(next() >> 11) * 0x1.0p-53L` | **Random Fraction** (0.0 to 1.0) | Takes the top 53 bits of a random integer and divides by $2^{53}$ to turn it into a decimal percentage. |
| **`rf2()`** | `rf() * 2.0L - 1.0L` | **Random Range** (-1.0 to 1.0) | Stretches the range: Doubles the fraction (0 to 2), then subtracts 1 (-1 to 1). |
| **`ri(n)`** | `next() % n` | **Random Choice** (0 to n-1) | Uses Modulo (%) to force a huge number into a small container size `n`. |
| **`gaussian()`** | Box-Muller Transform | **Bell Curve** | Uses `log` and `cos` on two random fractions to create numbers that cluster around 0 (small nudges) but occasionally go far (big jumps). |

## Visualizing the Engine
Think of `FastRNG` as a machine with a central engine (`next`) that powers different tools.

```mermaid
graph TD
    subgraph "1. The Engine (inside next())"
        State0[("s[0] (64-bit)")]
        State1[("s[1] (64-bit)")]
        
        %% The Result Calculation
        Sum["s[0] + s[1]"]
        RotResult["rotl(Sum, 17)"]
        Result[("RESULT (Raw Random Integer)")]
        
        %% Defining links explicitly to color them
        %% s0 paths (Red)
        State0 -- "1. Add" --> Sum
        State0 -- "2. Mix" --> XOR["s1 ^= s0"]
        
        %% s1 paths (Blue)
        State1 -- "1. Add" --> Sum
        State1 -- "2. Mix" --> XOR
        
        Sum --> RotResult --> Result
        
        %% The State Update
        NewS0["New s[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21)"]
        NewS1["New s[1] = rotl(s1, 28)"]
        
        XOR -- "3. Calc New States" --> NewS0
        XOR --> NewS1
        
        %% Feedback Loops
        NewS0 -. "4. Overwrite s[0]" .-> State0
        NewS1 -. "4. Overwrite s[1]" .-> State1
        
        %% Coloring the links
        %% 0, 1 are s0 outgoing (Red)
        linkStyle 0,1 stroke:#ff5555,stroke-width:2px,color:red;
        %% 2, 3 are s1 outgoing (Blue)
        linkStyle 2,3 stroke:#5555ff,stroke-width:2px,color:blue;
        %% 6 is NewS0 feedback (Red)
        linkStyle 6 stroke:#ff5555,stroke-width:2px,stroke-dasharray: 5 5,color:red;
        %% 7 is NewS1 feedback (Blue)
        linkStyle 7 stroke:#5555ff,stroke-width:2px,stroke-dasharray: 5 5,color:blue;
    end

    subgraph "2. The Tool (rf() helper)"
        Result --> Shift[">> 11 (Trim bits)"]
        Shift --> Scale["* 2^-53 (Scale to 0.0-1.0)"]
        FloatVal["u (Random Float 0.0-1.0)"]
        Scale --> FloatVal
    end

    subgraph "3. The Application (gaussian())"
        FloatVal -- "Need 2 of these" --> u1["u1 (Float 1)"]
        FloatVal -- "Need 2 of these" --> u2["u2 (Float 2)"]
        
        LogComp["sqrt(-2 * log(u1))"]
        CosComp["cos(2 * PI * u2)"]
        
        u1 --> LogComp
        u2 --> CosComp
        
        LogComp & CosComp --> Multiply["Multiply Together"]
        Final[("FINAL OUTPUT\n(Bell Curve Number)")]
        Multiply --> Final
    end
    
    style Result fill:#f96,stroke:#333,stroke-width:2px
    style Final fill:#6f9,stroke:#333,stroke-width:2px
    style State0 fill:#bbf,stroke:#333
    style State1 fill:#bbf,stroke:#333
```
