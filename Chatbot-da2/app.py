import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from flask import jsonify


from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json',encoding='utf-8').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # mã hóa mẫu - tách các từ thành mảng
    sentence_words = nltk.word_tokenize(sentence)
    # gốc từng từ - tạo dạng rút gọn cho từ
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# trả về mảng túi từ: 0 hoặc 1 cho mỗi từ trong túi tồn tại trong câu

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # túi từ - ma trận chữ N, ma trận từ vựng
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # gán 1 nếu từ hiện tại ở vị trí từ vựng
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # lọc ra các dự đoán dưới ngưỡng
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sắp xếp theo độ mạnh của xác suất
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    req_data = request.get_json()
    message = req_data['message']
    response = chatbot_response(message)
    result = {"response": response}
    return jsonify(result)
def chatbot_response(msg):
    ints = predict_class(msg, model)
    if len(ints) == 0:
        # Nếu không có dự đoán, trả về câu trả lời mặc định
        res = "Xin lỗi, tôi không hiểu câu hỏi của bạn."
    else:
        res = getResponse(ints, intents)
    return res


if __name__ == "__main__":
    app.run()