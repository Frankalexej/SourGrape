import pandas as pd
import numpy as np
import warnings

def check_missing_sd_columns(df):
    """
    Check that each feature column has a corresponding *_SD column.
    Warns if any SD column is missing.
    """
    # Identify feature and SD columns
    feature_cols = [col for col in df.columns if col != "phoneme" and not col.endswith("_SD")]
    sd_cols = {col: col for col in df.columns if col.endswith("_SD")}

    missing = []

    for feature in feature_cols:
        sd_name = feature + "_SD"
        if sd_name not in sd_cols:
            missing.append(sd_name)

    if missing:
        warnings.warn(
            f"The following features are missing SD columns: {missing}",
            UserWarning
        )

    return missing


def generate_phoneme_tokens(csv_path, phoneme_symbol, num_tokens, to_df=True):
    if isinstance(csv_path, str): 
        # Load the CSV file
        df = pd.read_csv(csv_path)
    elif isinstance(csv_path, pd.DataFrame): 
        df = csv_path
    else: 
        raise TypeError("csv_path must either be a string or pandas df! ")
    
    # Make sure the phoneme column is present
    if 'phoneme' not in df.columns:
        raise ValueError("The CSV must contain a 'phoneme' column.")
    
    # Optional: warn if SD columns are missing
    missing = check_missing_sd_columns(df)
    
    # Filter the dataframe for the given phoneme symbol
    phoneme_data = df[df['phoneme'] == phoneme_symbol]
    
    if phoneme_data.empty:
        raise ValueError(f"No data found for phoneme symbol: {phoneme_symbol}")
    
    # Identify feature columns (everything except phoneme and *_SD columns)
    feature_cols = [
        col for col in df.columns 
        if col != 'phoneme' and not col.endswith('_SD')
    ]
    
    # Get means for all features (vector)
    means = phoneme_data[feature_cols].iloc[0].to_numpy(dtype=float)
    
    # Build std vector aligned with feature_cols
    stds = []
    for feature in feature_cols:
        sd_col = feature + "_SD"
        if sd_col in phoneme_data.columns:
            stds.append(float(phoneme_data[sd_col].iloc[0]))
        else:
            stds.append(0.0)  # or raise, depending on how strict you want
    stds = np.array(stds, dtype=float)
    
    # Vectorized sampling: shape = (num_tokens, n_features)
    samples = np.random.normal(loc=means, scale=stds, size=(num_tokens, len(feature_cols)))
    
    # Build DataFrame
    df_features = pd.DataFrame(samples, columns=feature_cols)
    df_meta = pd.DataFrame({
        "phoneme": [phoneme_symbol] * num_tokens,
        "pid": np.arange(num_tokens, dtype=int),
    })
    
    result_df = pd.concat([df_meta, df_features], axis=1)
    
    if to_df:
        return result_df
    else:
        # preserve old behaviour: list of dicts
        return result_df.to_dict(orient="records")
    

def generate_words_from_csv(words_csv_path, features_csv_path, default_count=1):
    # Load inputs
    words_df = pd.read_csv(words_csv_path)
    if 'word' not in words_df.columns:
        raise ValueError("words.csv must have a 'word' column.")

    if isinstance(features_csv_path, str):
        features_df = pd.read_csv(features_csv_path)
    else:
        features_df = features_csv_path

    all_tokens = []
    current_cid = 0 # component id: one per phoneme

    # Loop *per word* (cannot avoid)
    for _, row in words_df.iterrows():
        word = row['word']
        count = row.get("number", default_count)

        phonemes = list(word)
        n_ph = len(phonemes)

        # For each word instance: wid = 0 .. count-1
        # Total phoneme tokens = count × n_ph
        total_tokens = count * n_ph

        # Expand wid: e.g., for count=3, n_ph=4 → [0,0,0,0,1,1,1,1,2,2,2,2]
        wid = np.repeat(np.arange(count), n_ph)

        # Expand positions (pos repeats for each wid)
        pos = np.tile(np.arange(n_ph), count)

        # Expand phoneme symbols
        expanded_phonemes = np.tile(phonemes, count)

        # Generate phoneme values in batch:
        # Group by distinct phonemes in this word
        df_list = []
        for ph in set(phonemes):
            n_ph_occ = (expanded_phonemes == ph).sum()
            ph_df = generate_phoneme_tokens(
                features_df,
                ph,
                n_ph_occ,
                to_df=True
            )
            ph_df["phoneme_symbol"] = ph
            df_list.append(ph_df)

        # Concatenate and restore order
        df_gen = pd.concat(df_list, ignore_index=True)

        # Shuffle back into original required sequence
        # (We sort by the order phonemes appear in expanded_phonemes)
        df_gen = df_gen.sort_values("phoneme_symbol")
        df_gen = df_gen.reset_index(drop=True)

        # Attach metadata
        df_gen["word"] = word
        df_gen["wid"] = wid
        df_gen["pos"] = pos
        df_gen["cid"] = np.arange(current_cid, current_cid + total_tokens)
        current_cid += total_tokens

        # Build identifier
        df_gen["identifier"] = (
            df_gen["word"] + "_" +
            df_gen["wid"].astype(str) + "_" +
            df_gen["pos"].astype(str) + "_" +
            df_gen["pid"].astype(str)
        )

        # Build word identifier
        df_gen["word_identifier"] = (
            df_gen["word"] + "_" +
            df_gen["wid"].astype(str)
        )

        # Drop helper column
        df_gen = df_gen.drop(columns=["phoneme_symbol"])

        all_tokens.append(df_gen)

    return pd.concat(all_tokens, ignore_index=True).drop(columns=["pid"])


if __name__ == "__main__": 
    features_path = "./features.csv"
    words_path = "./words.csv"
    tokens_path = "./tokens.csv"
    result_df = generate_words_from_csv(words_path, features_path, default_count=1000)
    result_df.to_csv(tokens_path, index=False)