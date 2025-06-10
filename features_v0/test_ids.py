import numpy as np
import pandas as pd

# --- Parameters ---
N = 10  # number of rows
M = 11  # number of samples per signal (each signal is an array of shape (M, 2))

# For reproducibility, set a random seed
np.random.seed(42)

# --- Create dummy data ---
# Generate random satellite and cell IDs (for example, 0-9) to allow duplicates and ordering tests.
ra_sat = np.random.randint(0, 4, size=N)
ra_cell = np.random.randint(0, 4, size=N)

# Generate dummy signals (each a (M, 2) array) with random float values.
# Using float32 to reduce memory usage.
samples = np.random.rand(N, M, 2).astype(np.float32)

# Create DataFrame – note that we use list(samples) so that each row stores the corresponding (M, 2) array.
df = pd.DataFrame({
    'ra_sat': ra_sat,
    'ra_cell': ra_cell,
    'samples': list(samples)
})

# Set the MultiIndex with levels 'ra_sat' and 'ra_cell' but keep the columns available (drop=False)
df.set_index(['ra_sat', 'ra_cell'], inplace=True, drop=False)

# --- Revised get_signals function ---
def get_signals(df, sat_id=None, cell_id=None):
    """
    Returns the signals associated with a satellite (sat_id), a cell (cell_id), or both.
    The filtering is done on a DataFrame that already contains 'ra_sat' and 'ra_cell' as columns,
    preserving the original row order.
    
    Args:
        df (pandas.DataFrame): DataFrame with a MultiIndex (ra_sat, ra_cell) and a 'samples' column.
        sat_id (int, optional): Satellite ID to filter by.
        cell_id (int, optional): Cell ID to filter by.
    
    Returns:
        Depending on the provided filters:
          - Both provided: a list of samples for the specific (sat_id, cell_id) combination.
          - Only sat_id: a list of (cell_id, samples) pairs.
          - Only cell_id: a list of (sat_id, samples) pairs.
          - None if no criteria or no matching data.
    """
    # If no filter is provided, return None.
    if sat_id is None and cell_id is None:
        return None

    # Use a temporary DataFrame that has 'ra_sat' and 'ra_cell' as columns.
    # If these columns already exist, no need to reset the index.
    if 'ra_sat' in df.columns and 'ra_cell' in df.columns:
        tmp = df
    else:
        tmp = df.reset_index()

    # If both sat_id and cell_id are provided, filter by both.
    if sat_id is not None and cell_id is not None:
        filtered = tmp[(tmp['ra_sat'] == sat_id) & (tmp['ra_cell'] == cell_id)]
        return filtered['samples'].tolist() if not filtered.empty else None

    # If only sat_id is provided, filter by sat_id and return (cell_id, samples) pairs.
    if sat_id is not None:
        filtered = tmp[tmp['ra_sat'] == sat_id]
        return list(zip(filtered['ra_cell'], filtered['samples'])) if not filtered.empty else None

    # If only cell_id is provided, filter by cell_id and return (sat_id, samples) pairs.
    if cell_id is not None:
        filtered = tmp[tmp['ra_cell'] == cell_id]
        return list(zip(filtered['ra_sat'], filtered['samples'])) if not filtered.empty else None


# --- Testing the get_signals function ---

# Test 1: Both sat_id and cell_id provided.
test_sat_id = 1
test_cell_id = 2
signals_both = get_signals(df, sat_id=test_sat_id, cell_id=test_cell_id)
print("Test 1: Both sat_id and cell_id provided")
if signals_both is not None:
    print(f"Number of signals for sat_id {test_sat_id} and cell_id {test_cell_id}: {len(signals_both)}")
else:
    print("No signals found for the given (sat_id, cell_id) combination.")

# Test 2: Only sat_id provided.
signals_sat = get_signals(df, sat_id=test_sat_id)
print("\nTest 2: Only sat_id provided")
if signals_sat is not None:
    print(f"Number of signals for sat_id {test_sat_id}: {len(signals_sat)}")
    # Show the first 5 pairs (cell_id, samples) to verify order is preserved.
    print("First 5 (cell_id, samples) pairs:")
    for pair in signals_sat[:5]:
        print(pair[0])
else:
    print("No signals found for the given sat_id.")

# Test 3: Only cell_id provided.
signals_cell = get_signals(df, cell_id=test_cell_id)
print("\nTest 3: Only cell_id provided")
if signals_cell is not None:
    print(f"Number of signals for cell_id {test_cell_id}: {len(signals_cell)}")
    # Show the first 5 pairs (sat_id, samples) to verify order is preserved.
    print("First 5 (sat_id, samples) pairs:")
    for pair in signals_cell[:5]:
        print(pair[0])
else:
    print("No signals found for the given cell_id.")

# Optional: Verify order by comparing with a direct boolean filtering on the reset DataFrame.
# --- Verification: Use the existing DataFrame without resetting the index ---
if 'ra_sat' in df.columns and 'ra_cell' in df.columns:
    df_verif = df  # Already has the necessary columns.
else:
    df_verif = df.reset_index()

print("\nVerification: First 5 entries for sat_id", test_sat_id, "from the DataFrame:")
print(df_verif[df_verif['ra_cell'] == test_cell_id].head())


print (ra_sat)

print(ra_cell)

print(samples)