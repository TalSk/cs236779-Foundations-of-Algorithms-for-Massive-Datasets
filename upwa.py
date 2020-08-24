import numpy as np
from utils import CFMethod, _split_train_test, RatingData, load_movielens_100k, load_jester_100k


def pearson_correlation(u_ratings: np.ndarray, v_ratings: np.ndarray) -> float:
    """
    Calculates the Pearson correlation between two users' rating vectors.
    :param u_ratings: First users' rating vector (M,)
    :param v_ratings: Second users' rating vector (M,)
    :return: Pearson correlation values, where -1 indicates absolute negative correlation, 0 no correlation
             and 1 absolute positive correlation.
    """
    assert (u_ratings.size == v_ratings.size)
    # First all co-rated items, while calculating the averages rating for them.
    aru = 0
    arv = 0
    co_rated_indices = []
    for i in range(len(u_ratings)):
        if not np.isnan(u_ratings[i]) and not np.isnan(v_ratings[i]):
            aru += u_ratings[i]
            arv += v_ratings[i]
            co_rated_indices += [i]
    if len(co_rated_indices) == 0:
        return 0.
    aru /= len(co_rated_indices)
    arv /= len(co_rated_indices)

    # Calculate numerator and denominator according to formula.
    numerator = sum([(u_ratings[i] - aru) * (v_ratings[i] - arv) for i in co_rated_indices])
    denominator = np.sqrt(sum([(u_ratings[i] - aru) ** 2 for i in co_rated_indices])) * np.sqrt(
        sum([(v_ratings[i] - arv) ** 2 for i in co_rated_indices]))
    if denominator == 0:
        # Indicates a case where the average rating of a user is the same as all their rating (or that they have no
        # ratings), either way, there can't be a correlation.
        return 0.
    return numerator / denominator


class UPWA(CFMethod):
    """
    Class for the User-based, Pearson correlation, weighted average sum CF method.
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.rating_matrix = None
        self.weights = {}
        self.N = None
        self.M = None

    def train(self, ratings: RatingData) -> None:
        """
        Train the model using the given matrix. In this current implementation, simply saves the matrix in order to be
        more efficient and not calculate all similarities here.
        :param ratings: The user ratings dataset.
        """
        self.rating_matrix = ratings.as_matrix()
        self.N, self.M = self.rating_matrix.shape

    def predict(self, user: int, item: int) -> float:
        """
        Predicts how much would user @user rate item @item.
        :param user: The user to predict for.
        :param item: The item to predict about.
        :return: Value indicating interest of user in item, based on the weighted average sum
                 (can be a float even for binary cases)
        """
        assert (user < self.N and item < self.M)
        # If this user had already rated given item, just return this rating.
        if not np.isnan(self.rating_matrix[user][item]):
            return self.rating_matrix[user][item]

        # Find all users who had rated given item.
        users_rated = []
        for v in range(self.N):
            if v != user and not np.isnan(self.rating_matrix[v][item]):
                users_rated += [v]

        # Calculate similarities (AKA weights) between given user and all other users who rated this item.
        self.weights = {}
        for v in users_rated:
            # Skip user and users who had already had been computed.
            if v == user or str(sorted([user, v])) in self.weights:
                continue
            # We remember weights as a string of a sorted list containing users (since the weight is symmetric).
            self.weights[str(sorted([user, v]))] = pearson_correlation(self.rating_matrix[user], self.rating_matrix[v])

        # Get top neighbors in terms of similarities
        neighbors = sorted([(self.weights[str(sorted([user, v]))], v) for v in users_rated], reverse=True)[
                    :self.n_neighbors]

        # Calculate denominator and numerator according to formula.
        denominator = 0
        numerator = 0
        for sim, v in neighbors:
            denominator += np.fabs(sim)
            masked = np.ma.array(self.rating_matrix[v], mask=False)
            masked.mask[item] = True
            numerator += (self.rating_matrix[v][item] - np.nanmean(masked).item()) * sim
        if denominator == 0:
            return 0.

        pai = np.nanmean(self.rating_matrix[user]).item()
        pai += numerator / denominator
        return pai


def test_example():
    # Running example matrix, as defined in the paper.
    example_dataset = RatingData(5, 4)
    example_dataset.add(0, 0, 4)
    example_dataset.add(0, 2, 5)
    example_dataset.add(0, 3, 5)
    example_dataset.add(1, 0, 4)
    example_dataset.add(1, 1, 2)
    example_dataset.add(1, 2, 1)
    example_dataset.add(2, 0, 3)
    example_dataset.add(2, 2, 2)
    example_dataset.add(2, 3, 4)
    example_dataset.add(3, 0, 4)
    example_dataset.add(3, 1, 4)
    example_dataset.add(4, 0, 2)
    example_dataset.add(4, 1, 1)
    example_dataset.add(4, 2, 3)
    example_dataset.add(4, 3, 5)

    upwa = UPWA(n_neighbors=3)
    upwa.train(example_dataset)

    print(upwa.predict(0, 1))  # Should be approximately 3.95


def test_movielens_100k():
    # Load MovieLens-100k database into a rating matrix. Reserving 1% for estimation.
    ratings = load_movielens_100k()
    train, test = _split_train_test(ratings)

    # Create model and pass the training rating matrix.
    upwa = UPWA(n_neighbors=5)
    upwa.train(train)

    # Calculate mean absolute error on all test ratings.
    mae = 0
    rmse = 0
    for _, t in enumerate(test):
        u, i, r = t.user, t.item, t.rating
        r_pred = upwa.predict(u, i)
        mae += abs(r_pred - r)
        rmse += (r_pred - r) ** 2
        if (_ % 10) == 0:
            print(_)
    mae /= len(test)
    rmse /= len(test)
    rmse = np.sqrt(rmse)
    print(f"Mean absolute error for the learnt MovieLens-100k model: {mae}. Root mean squared error: {rmse}")


def test_jester_100k():
    # Load Jester-100k database into a rating matrix. Reserve 1% for estimation
    ratings = load_jester_100k()
    train, test = _split_train_test(ratings)

    # Create model and pass the training rating matrix.
    upwa = UPWA(n_neighbors=5)
    upwa.train(train)

    # Calculate mean absolute error on all test ratings.
    mae = 0
    rmse = 0
    for _, t in enumerate(test):
        u, i, r = t.user, t.item, t.rating
        r_pred = upwa.predict(u, i)
        mae += abs(r_pred - r)
        rmse += (r_pred - r) ** 2
        if (_ % 10) == 0:
            print(_)
    mae /= len(test)
    rmse /= len(test)
    rmse = np.sqrt(rmse)
    print(f"Mean absolute error for the learnt Jester-100k model: {mae}. Root mean squared error: {rmse}")


def main():
    # TODO: Uncomment depending on the chosen experiment.
    # test_example()
    # test_movielens_100k()
    # test_jester_100k()
    pass


if __name__ == '__main__':
    main()
