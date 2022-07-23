from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import config
from trainer_utils import *
import os.path

# start bot
def start(update: Update, context):
    context.message.reply_text(
        "Добрый день! Вы можете написать мне что бы узнать о \
        процентной ставке и первоначальном взносе для интерисующией вас ипотеки!")


# generate answer to the user
def get_mortgage(update: Update, context):
    inp = '%s' % context.message.text
    input_data = bag_of_words(inp, words)
    results = model.predict([input_data])
    results_index = np.argmax(results)
    tag = labels[results_index]

    # check prediction confidence
    if max(results[0]) > 0.6:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                name = tg['mortgageName']
                rate = tg['rate']
                initial_fee = tg['initial_fee']
                url = tg['url']
        text = [name, "- процентная ставка: ", rate, \
                ", первоначальный взнос: ", initial_fee, "подробнее можно узнать по ссылке: ", url]
        message = " ".join(str(term) for term in text)
        context.message.reply_text(message)

    else:
        context.message.reply_text("""Извините, я не совсем вас понимаю, попробуйте перефразировать свой запрос.
        Также вы можете ознакомиться с возможными вариантами ипотеки на сайте банка https://www.sberbank.ru/ru/person/credits/homenew""")

# show list of functions
def help(update: Update, context):
    context.message.reply_text("""Доступные команды :-
    /help - Что бы увидеть все доступные команды
    /site - Получить ссылку сайт банка""")


# show website
def site(update: Update, context):
    context.message.reply_text("https://www.sberbank.ru/ru/person/credits/homenew")


#in case if /command is invalid show message to the user
def unknown(update: Update, context):
    context.message.reply_text(
        "Извините, команда '%s' на данный момент не предусмотрена" % context.message.text)


# set model parameters and load the model
data = extract_data()
words, labels, docs_x, docs_y = intents_prepare(data)
training, output = input_prepare(words, labels, docs_x, docs_y)
net = net_setting(len(training[0]), len(output[0]))
#load model if exits, else train and save
if os.path.exists("model.tflearn.index"):
    model = load_model(training, output)
else:
    model = train_model(training, output, net)

updater = Updater(config.token)
updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('help', help))
updater.dispatcher.add_handler(CommandHandler('site', site))

# Filters out unknown commands
updater.dispatcher.add_handler(MessageHandler(Filters.command, unknown))
# Filters out unknown messages.
updater.dispatcher.add_handler(MessageHandler(Filters.text, get_mortgage))

updater.start_polling()
