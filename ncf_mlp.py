import numpy as np

from utils import CFMethod, load_movielens_1m, load_movielens_100k, load_jester_100k, _split_train_test, RatingData, \
    UserRating
from time import time

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from typing import List

DO_TRAIN_1M = False


class NCF_MLP(CFMethod):
    def __init__(self, num_users: int, num_items: int, model_out_file, layers: list, num_outputs: int = 1,
                 negative_samples: List[List] = None, learning_rate: int = 0.001, batch_size: int = 256,
                 early_stop: int = 8, activation: str = 'sigmoid', loss: str = 'binary_crossentropy',
                 sampled_negatives: int = 2, do_evaluation=True):
        self.N = num_users
        self.M = num_items
        self.batch_size = batch_size
        self.negative_samples = negative_samples
        self.early_stop = early_stop
        self.do_evaluation = do_evaluation
        self.model_out_file = model_out_file
        self.model = self._build_model(self.N, self.M, layers, num_outputs, activation)
        self.model.compile(optimizer=Adam(lr=learning_rate), loss=loss)
        self.model.summary()

        self.top_k = 10
        self.sampled_negatives = sampled_negatives
        self.evaluated_negatives = 100

    @staticmethod
    def _build_model(users: int, items: int, layers: list, num_outputs: int, activation: str):
        user = Input(shape=(1,), dtype='int32', name='user')
        item = Input(shape=(1,), dtype='int32', name='item')

        user_embedding = Embedding(input_dim=users, output_dim=layers[0] // 2, name='user_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(l2=0))
        item_embedding = Embedding(input_dim=items, output_dim=layers[0] // 2, name='item_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(l2=0))

        user_vector = Flatten()(user_embedding(user))
        item_vector = Flatten()(item_embedding(item))

        # Concat so we get target dimension
        full_vector = Concatenate()([user_vector, item_vector])

        for num, val in enumerate(layers):
            cur = Dense(val, activation='relu', name=f'layer_{num}', kernel_regularizer=l2(l2=0))
            full_vector = cur(full_vector)

        out = Dense(num_outputs, activation=activation, name='out')(full_vector)
        return Model(inputs=[user, item], outputs=out)

    def predict(self, user: int, item: int) -> int:
        pred = self.model.predict([np.array([user]), np.array([item])], batch_size=1, verbose=0)
        return pred[0][0]

    def _evaluate_one(self, x: UserRating) -> int:
        user, item, rating = x.user, x.item, x.rating
        # Get negative examples for this user
        neg_items = self.negative_samples[user]
        items = neg_items
        np.random.shuffle(items)
        items_shuffle = np.array(items)[:self.evaluated_negatives]
        items_shuffle = np.append(items_shuffle, [item])
        # Predict for these items
        users = np.full(len(items_shuffle), user, dtype='int32')
        pred = self.model.predict([users, items_shuffle], batch_size=len(items_shuffle), verbose=0)

        # Check that the item is within top 10 of results
        results = sorted([(pred[i][0], items_shuffle[i]) for i in range(len(items_shuffle))], reverse=True)
        top_10 = [x[1] for x in results[:self.top_k]]
        if item in top_10:
            # Hit
            return 1
        return 0

    def _evaluate(self, valid: RatingData) -> float:
        hits = 0
        for x in valid:
            hits += self._evaluate_one(x)

        return hits / len(valid)

    def train(self, ratings: RatingData, epochs: int = 100, print_every: int = 1) -> None:
        train, valid = _split_train_test(ratings, 0.1)

        best_hr, best_epoch, num_epochs_no_improve = 0, -1, 0
        user_pos, item_pos, ratings_pos = train.as_lists()
        for epoch in range(epochs):
            user_neg, item_neg, ratings_neg = train.generate_negatives(self.sampled_negatives)

            t1 = time()
            h = self.model.fit([np.array(user_pos + user_neg), np.array(item_pos + item_neg)],
                               np.array(ratings_pos + ratings_neg),
                               batch_size=self.batch_size, workers=5, shuffle=True)
            t2 = time()
            if epoch % print_every == 0 and self.do_evaluation:
                hr = self._evaluate(valid)
                print(f'Epoch {epoch} [{t2 - t1} s]: HR = {hr}')
                if hr > best_hr:
                    best_hr = hr
                    best_epoch = epoch
                    self.model.save_weights(self.model_out_file, overwrite=True)
                    print('Saved!')
                    num_epochs_no_improve = 0
                else:
                    num_epochs_no_improve += 1
                    if num_epochs_no_improve > self.early_stop:
                        break
        print(f'Done. Best epoch: {best_epoch}, best HR = {best_hr}')


def _generate_negatives(rating_matrix: np.ndarray, num_users: int, num_items: int, neg_file: str) -> None:
    """
    For each user, generates 99 random items that don't have a rating for, to a file.

    :param rating_matrix: A NxM rating matrix, containing
    :param num_users: N, the number of users in the dataset.
    :param num_items: M, the number of items in the dataset.
    :param neg_file: Path to generated file.
    """
    with open(neg_file, 'w') as f:
        for u in range(0, num_users):
            negatives = []
            while len(negatives) < 99:
                temp = np.random.randint(0, num_items)
                if temp not in negatives and np.isnan(rating_matrix[u][temp]):
                    negatives += [str(temp)]
            f.write(f'{u}\t' + '\t'.join(negatives) + '\n')


def _predict_test(class_instance: NCF_MLP, users: List, items: List):
    """
    Predicts ratings using trained model for pairs of users and items.
    :param class_instance: An instance of NCF_MLP containing a trained model.
    :param users: A list containing user ids to predict for (in order like items).
    :param items: A list containing item ids to predict for (in order like users).
    :return:
    """
    return class_instance.model.predict([np.array(users), np.array(items)], verbose=1)


def _calculate_mae_rmse(ground_truth: List, pred: List):
    """
    Calculates Mean Absolute
    :param ground_truth:
    :param pred:
    :return:
    """
    mae = 0
    rmse = 0
    for i in range(len(ground_truth)):
        mae += abs(ground_truth[i] - pred[i])
        rmse += (ground_truth[i] - pred[i]) ** 2
    mae /= len(ground_truth)
    rmse /= len(ground_truth)
    rmse = np.sqrt(rmse)
    return mae[0], rmse[0]


def test_movielens_1m_binary():
    # Load ratings dataset
    ratings = load_movielens_1m()

    train_ratings, _ = _split_train_test(ratings, 50)
    _, test_ratings = _split_train_test(_, 10)

    # Uncomment to re-generate the negative samples.
    # _generate_negatives(rating_matrix=ratings.as_matrix(), num_users=ratings.N, num_items=ratings.M,
    #                     neg_file=r"datasets\ml-1m\neg.dat")

    # Load negative dataset
    negatives = []
    with open(r"datasets\ml-1m\neg.dat", "r") as f:
        for x, line in enumerate(f):
            _line = [int(x) for x in line.strip().split("\t")]
            temp = []
            for item in _line[1:]:
                temp += [item]
            negatives += [temp]

    ncf_mlp = NCF_MLP(num_users=ratings.N, num_items=ratings.M, negative_samples=negatives,
                      model_out_file=r"models\ml_1m_64.model", layers=[256, 128, 64])

    if DO_TRAIN_1M:
        ncf_mlp.train(train_ratings)
    else:
        ncf_mlp.model.load_weights(r"models\ml_1m_64.model")

    t_users, t_items, t_ratings = test_ratings.as_lists()
    preds = _predict_test(ncf_mlp, t_users, t_items)

    mae, rmse = _calculate_mae_rmse(t_ratings, preds)
    print(f"Mean absolute error for the learnt MovieLens-1M model: {mae}. Root mean squared error: {rmse}")


def test_movielens_100k():
    # Load ratings dataset
    ratings = load_movielens_100k()
    train_ratings, test_ratings = _split_train_test(ratings, 10)
    ncf_mlp = NCF_MLP(num_users=ratings.N, num_items=ratings.M, model_out_file=r"models\ml_100k.model",
                      layers=[64, 32, 16, 8], num_outputs=1, sampled_negatives=0,
                      do_evaluation=False)
    ncf_mlp.train(train_ratings, epochs=5)

    t_users, t_items, t_ratings = test_ratings.as_lists()
    preds = _predict_test(ncf_mlp, t_users, t_items)

    mae, rmse = _calculate_mae_rmse(t_ratings, preds)
    print(f"Mean absolute error for the learnt MovieLens-100k model: {mae}. Root mean squared error: {rmse}")


def test_jester_100k():
    # Load ratings dataset
    ratings = load_jester_100k()
    train_ratings, test_ratings = _split_train_test(ratings, 10)
    ncf_mlp = NCF_MLP(num_users=ratings.N, num_items=ratings.M, model_out_file=r"models\jester_100k.model",
                      layers=[64, 32, 16, 8], num_outputs=1, sampled_negatives=0,
                      do_evaluation=False)
    ncf_mlp.train(train_ratings, epochs=5)

    t_users, t_items, t_ratings = test_ratings.as_lists()
    preds = _predict_test(ncf_mlp, t_users, t_items)

    mae, rmse = _calculate_mae_rmse(t_ratings, preds)
    print(f"Mean absolute error for the learnt Jester-100k model: {mae}. Root mean squared error: {rmse}")


def main():
    # TODO: Uncomment depending on the chosen experiment.
    # test_movielens_1m_binary()
    # test_movielens_100k()
    # test_jester_100k()
    pass


if __name__ == '__main__':
    main()
