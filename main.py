import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention, Concatenate
from tensorflow.keras.optimizers import Adam


print ("\n" + "=" * 70)
print ("        	FOOTBALL PLAYER TRAJECTORY PREDICTION")
print ("=" * 70)


#1.LOAD DATA

print ("\n[1] LOADING MATCH TRACKING DATA")
print ("-" * 70)

home = pd.read_csv("Sample_Game_1_RawTrackingData_Home_Team.csv",skiprows=[0, 1])

away = pd.read_csv("Sample_Game_1_RawTrackingData_Away_Team.csv",skiprows=[0, 1])

print ("Home team rows :", len(home))
print ("Away team rows :", len(away))
print ("Home columns   :", len(home.columns))
print ("Away columns   :", len(away.columns))


#2.FIND PLAYER 11

print ("\n[2] SELECTING PLAYER 11 TRAJECTORY")
print ("-" * 70)

home_columns = list(home.columns)

player11_index = home_columns.index("Player11")

player11_x_col = home_columns[player11_index]
player11_y_col = home_columns[player11_index + 1]

print ("X coordinate column :", player11_x_col)
print ("Y coordinate column :", player11_y_col)

print ("\nFirst 5 Player 11 positions:")

print (home[[player11_x_col, player11_y_col]].head().to_string(index=False))


#3.COMBINE HOME + AWAY

print ("\n[3] COMBINING BOTH TEAMS")
print ("-" * 70)

data = pd.concat(
    [
        home.iloc[:, :3],
        home.iloc[:, 3:],
        away.iloc[:, 3:]
    ],axis=1)

print ("Combined dataset shape:", data.shape)

#4.NUMERICAL CONVERSION

print ("\n[4] PREPROCESSING")
print ("-" * 70)

data = data.apply(pd.to_numeric, errors="coerce")

missing_before = data.isna().sum().sum()

print ("Missing values before interpolation:", missing_before)

data = data.interpolate()
data = data.ffill()
data = data.bfill()

missing_after = data.isna().sum().sum()

print ("Missing values after preprocessing :", missing_after)

print ("Preprocessing completed.")

#5.TARGET

print ("\n[5] DEFINING PREDICTION TARGET")
print ("-" * 70)

target_x_index = 3 + (player11_index - 3)
target_y_index = target_x_index + 1

target_x_col = data.columns[target_x_index]
target_y_col = data.columns[target_y_index]

print ("Target X:", target_x_col)
print ("Target Y:", target_y_col)

target = data.iloc[:, [target_x_index, target_y_index]].values

print ("Target shape:", target.shape)

print ("\nExample target positions:")
print (target[:5])

#6. INPUT FEATURES

print ("\n[6] DEFINING INPUT FEATURES")
print ("-" * 70)

features = data.iloc[:, 3:].values

print ("Number of tracking features:", features.shape[1])
print ("Number of frames:", features.shape[0])

print ("\nEach frame contains:")
print ("Player positions + ball position")
print ("These values are used as input to the LSTM.")

# 7. NORMALIZATION

print ("\n[7] NORMALIZING TRACKING DATA")
print ("-" * 70)

scaler = MinMaxScaler()

features = scaler.fit_transform(features)

print ("Normalization completed.")

print ("\nExample normalized values:")
print (features[0, :10])

#8.TRAIN / TEST SPLIT

print ("\n[8] TRAIN / TEST SPLIT")
print ("-" * 70)

split = int(len(features) * 0.70)

train_features = features[:split]
test_features = features[split:]

train_target = target[:split]
test_target = target[split:]

print ("Training frames:", len(train_features))
print ("Testing frames :", len(test_features))

print ("Training percentage: 70%")
print ("Testing percentage : 30%")

print ("\nChronological split used.")
print ("Past data → Training")
print ("Future data → Testing")

#9.CREATE SEQUENCES

print ("\n[9] CREATING TIME-SERIES SEQUENCES")
print ("-" * 70)

input_len = 25
output_len = 10
stride = 5

print ("Input window :", input_len, "frames")
print ("Prediction   :", output_len, "future frames")
print ("Stride       :", stride)

print ("\nMeaning:")
print ("The model observes 25 previous frames")
print ("and predicts Player 11's X,Y position for the next 10 frames.")

def create_sequences(features, target):

    X = []
    y = []

    for i in range(0,len(features) - input_len - output_len + 1,stride):

        X.append(features[i:i + input_len])
        y.append(target[i + input_len:i + input_len + output_len].flatten())

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_features,train_target)

X_test, y_test = create_sequences(test_features,test_target)

print ("\nSequence creation completed.")

print ("X_train shape:", X_train.shape)
print ("y_train shape:", y_train.shape)

print ("X_test shape :", X_test.shape)
print ("y_test shape :", y_test.shape)

#10.UNDERSTAND ONE TRAINING SAMPLE

print ("\n[10] UNDERSTANDING ONE TRAINING SAMPLE")
print ("-" * 70)

print ("One input sample:")
print ("25 frames ×", X_train.shape[2], "features")

print ("\nInput shape:")
print (X_train[0].shape)

print ("\nTarget shape:")
print (y_train[0].shape)

print ("\nTarget contains:")
print ("10 future frames × 2 coordinates = 20 values")

print ("\nFirst future X,Y positions:")

sample_target = y_train[0].reshape(output_len, 2)

for i, position in enumerate(sample_target):
    print ("Future frame",i + 1,"→ X =",round(position[0], 4),"Y =",round(position[1], 4))

#11.STANDARD LSTM

print ("\n" + "=" * 70)
print ("[11] BUILDING STANDARD LSTM")
print ("=" * 70)

model1 = Sequential([
    Input(shape=(input_len, X_train.shape[2])),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(output_len * 2)])

model1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse")

model1.summary()

#12.TRAIN STANDARD LSTM

print ("\n[12] TRAINING STANDARD LSTM")
print ("-" * 70)

history1 = model1.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1)

#13.STANDARD LSTM PREDICTION

print ("\n[13] STANDARD LSTM PREDICTION")
print ("-" * 70)

pred1 = model1.predict(X_test)

print ("Prediction shape:", pred1.shape)

print ("\nExample prediction:")
print (pred1[0].reshape(output_len, 2))

#14.ATTENTION LSTM

print ("\n" + "=" * 70)
print ("[14] BUILDING ATTENTION LSTM")
print ("=" * 70)

inputs = Input(shape=(input_len, X_train.shape[2]))

x = LSTM(64,return_sequences=True)(inputs)

x = LSTM(64,return_sequences=True)(x)

attention = Attention()([x, x])

x = Concatenate()([
    x[:, -1, :],
    attention[:, -1, :]])

outputs = Dense(output_len * 2)(x)

model2 = Model(
    inputs,
    outputs)

model2.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse")

model2.summary()

#15.TRAIN ATTENTION LSTM

print ("\n[15] TRAINING ATTENTION LSTM")
print ("-" * 70)

history2 = model2.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1)

#16.ATTENTION LSTM PREDICTION

print ("\n[16] ATTENTION LSTM PREDICTION")
print ("-" * 70)

pred2 = model2.predict(X_test)

print ("Prediction shape:", pred2.shape)

print ("\nExample prediction:")
print (pred2[0].reshape(output_len, 2))

#17.COMPARE ONE TRAJECTORY

print ("\n" + "=" * 70)
print ("[17] ACTUAL VS PREDICTED TRAJECTORY")
print ("=" * 70)

actual = y_test[0].reshape(output_len, 2)

standard = pred1[0].reshape(output_len, 2)

attention_pred = pred2[0].reshape(output_len, 2)

print ("\nFrame       Actual X   Actual Y   LSTM X     LSTM Y     Attn X     Attn Y")

for i in range(output_len):

    print (
        f"{i + 1:5d}"
        f"{actual[i,0]:12.4f}"
        f"{actual[i,1]:11.4f}"
        f"{standard[i,0]:11.4f}"
        f"{standard[i,1]:11.4f}"
        f"{attention_pred[i,0]:11.4f}"
        f"{attention_pred[i,1]:11.4f}")

#18.CALCULATE ERRORS

print ("\n" + "=" * 70)
print ("[18] MODEL EVALUATION")
print ("=" * 70)

mae1 = mean_absolute_error(y_test,pred1)

rmse1 = np.sqrt(mean_squared_error(y_test,pred1))

mae2 = mean_absolute_error(y_test,pred2)

rmse2 = np.sqrt(mean_squared_error(y_test,pred2))


print ("\nStandard LSTM")
print ("MAE :", round(mae1, 6))
print ("RMSE:", round(rmse1, 6))

print ("\nAttention LSTM")
print ("MAE :", round(mae2, 6))
print ("RMSE:", round(rmse2, 6))

#19.IMPROVEMENT

mae_improvement = ((mae1 - mae2) / mae1) * 100

rmse_improvement = ((rmse1 - rmse2) / rmse1) * 100

print ("\n" + "=" * 70)
print ("[19] IMPROVEMENT WITH ATTENTION")
print ("=" * 70)

print ("MAE improvement :",round(mae_improvement, 2),"%")

print ("RMSE improvement:",round(rmse_improvement, 2),"%")

#20.FINAL INTERPRETATION

print ("\n" + "=" * 70)
print ("[20] FINAL INTERPRETATION")
print ("=" * 70)

print ("""
The model observes previous player and ball movements
and learns the temporal movement pattern.

The Standard LSTM uses its learned sequence representation
to predict Player 11's future X,Y coordinates.

The Attention-LSTM additionally learns which information
from the input sequence is more useful for the prediction.

The two models are evaluated on the same unseen test data.
Lower MAE and RMSE indicate more accurate trajectory prediction.""")

if mae2 < mae1 and rmse2 < rmse1:

    print ("RESULT:")
    print ("Attention-LSTM performs better than Standard LSTM.")

elif mae1 < mae2 and rmse1 < rmse2:

    print ("RESULT:")
    print ("Standard LSTM performs better than Attention-LSTM.")

else:

    print ("RESULT:")
    print ("The models have mixed performance.")


print ("\n" + "=" * 70)
print ("                    EXPERIMENT COMPLETE")
print ("=" * 70)
