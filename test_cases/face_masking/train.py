from sklearn.model_selection import train_test_split
from data_preparation import load_and_preprocess_data, prepare_data, generate_tensor_data
from model import create_model


def train_model():
	train = load_and_preprocess_data()
	X, Y = prepare_data(train)
	xtrain, xval, ytrain, yval = train_test_split(X, Y, train_size=0.8, random_state=0)
	tensordata = generate_tensor_data(xtrain)
	model = create_model()
	history = model.fit(
		tensordata.flow(xtrain, ytrain, batch_size=32),
		steps_per_epoch=len(xtrain) // 32,
		epochs=3,
		# epochs=50,
		verbose=1,
		validation_data=(xval, yval)
	)
	return history


if __name__ == "__main__":
	print("Training started...")
	history = train_model()
	print("Training completed.")
