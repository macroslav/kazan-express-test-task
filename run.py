from pathlib import Path
import logging
import argparse
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.metrics.metrics import hierarchical_f1_score
from src.data_loaders.data_loaders import load_data
from src.data_preprocessors.preprocessors import (preprocess_data, fill_description_nans, get_categories_to_change,
                                                  change_category_to_parent, process_categories_data,
                                                  extract_text_features)

MODELS_SAVE_DIR = Path('models')


def run_pipeline(save_model: bool):
    logger.info("Loading data")
    data_dir = Path('data/raw')
    train_data, test_data, categories_data = load_data(data_dir=data_dir)

    logger.info("Processing data")
    train_data['new_category'] = train_data['category_id']
    while len(get_categories_to_change(train_data)) > 10:
        categories_to_change = get_categories_to_change(train_data)
        train_data = change_category_to_parent(categories_to_change=categories_to_change,
                                               data=train_data,
                                               categories_data=categories_data)
    train_data = train_data[train_data.new_category != 1]
    train_data = fill_description_nans(train_data)
    test_data = fill_description_nans(test_data)

    train_data = train_data.drop_duplicates(subset=['title', 'short_description', 'category_id'])

    used_categories: list[int] = train_data.new_category.unique().tolist()
    categories_data = process_categories_data(categories_data=categories_data,
                                              used_categories=used_categories)

    drop_columns = ['short_description', 'title', 'rating', 'feedback_quantity', 'name_value_characteristics']
    train_data = preprocess_data(train_data, drop_columns)
    test_data = preprocess_data(test_data, drop_columns)

    train_features, test_features = extract_text_features(train_data, test_data)

    labels_data = train_data[['category_id', 'new_category']]

    x_train, x_val, y_train, y_val = train_test_split(train_features,
                                                      labels_data,
                                                      test_size=0.3,
                                                      shuffle=True,
                                                      random_state=100)
    logger.info(f"X_train shape: {x_train.shape}")

    logger.info("Fitting model")
    model = LogisticRegression(C=1.0)
    model.fit(x_train, y_train[['new_category']])

    if save_model:
        logger.info("Save model to .pickle")
        with open(MODELS_SAVE_DIR / 'model.pickle', 'wb') as f:
            pickle.dump(model, f)

    logger.info("Getting predicts")
    val_predicts = model.predict(x_val)
    decoder = dict(categories_data.loc[:, ['id', 'full_path_id']].values)

    val_predicts_paths = [decoder[pred] for pred in val_predicts]
    val_true_paths = [decoder[pred] for pred in y_val.category_id.values]
    logger.info(f"Validation f1 score: {hierarchical_f1_score(predicts=val_predicts_paths, true=val_true_paths)}")

    train_predicts = model.predict(x_train)
    train_predicts_paths = [decoder[pred] for pred in train_predicts]
    train_true_paths = [decoder[pred] for pred in y_train.category_id.values]
    logger.info(f"Validation f1 score: {hierarchical_f1_score(predicts=train_predicts_paths, true=train_true_paths)}")

    logger.info("Predict on test data")
    predicts_final = model.predict(test_features)
    test_data['predicted_category_id'] = predicts_final
    submission = test_data[['id', 'predicted_category_id']]
    submission.to_parquet('result.parquet')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Parser of arguments')
    parser.add_argument('--save_model',
                        type=bool,
                        default=True,
                        help=''' Flag for saving model ''')
    args = parser.parse_args()
    run_pipeline(args.save_model)
