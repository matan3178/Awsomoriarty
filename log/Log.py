class Log:
    log = list()

    def print(self, text):
        self.log.append(text)
        print(text)
        return

    def print_log(self):
        print(self.log)
        return