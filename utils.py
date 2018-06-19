from math import floor
from time import time
from smtplib import SMTP
from email.message import EmailMessage
from collections import Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Timer:
    """Simple Timer object which prints elapsed time since its creation"""

    def __init__(self, name):
        self.name = name
        self.__start_time = None
        self.__end_time = None
        self.start()

    def start(self):
        self.__start_time = time()

    def stop(self):
        self.__end_time = time()
        self.__get_elapsed__()

    def __get_elapsed__(self):
        """function to return correctly formatted string according to time units"""
        elapsed = (self.__end_time - self.__start_time)
        unit = "seconds"
        if elapsed >= 3600:
            unit = "minutes"
            hours = elapsed / 3600
            minutes = hours % 60
            hours = floor(hours)
            print("{} took {} hours and {:.2f} {} to complete".format(self.name, hours, minutes, unit))
        elif elapsed >= 60:
            minutes = floor(elapsed / 60)
            seconds = elapsed % 60
            print("{} took {} minutes and {:.2f} {} to complete".format(self.name, minutes, seconds, unit))
        else:
            print("{} took {:.2f} {} to complete".format(self.name, elapsed, unit))


def sendEmail(message):
    """utility function to send an email with results from a training run"""
    message_string = '\n'.join(message)
    recipients = ['nadavo@campus.technion.ac.il', 'olegzendel@campus.technion.ac.il']
    msg = EmailMessage()
    msg['Subject'] = 'Finished training and predicting DependencyParser'
    msg['From'] = 'someserver@technion.ac.il'
    msg['To'] = ', '.join(recipients)
    msg.set_content(message_string)
    sender = SMTP('localhost')
    sender.send_message(msg)
    sender.quit()
