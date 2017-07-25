class Log:
    log = list()

    def print(self, text):
        self.log.append(text)
        print(text)
        return

    def print_log(self):
        for line in self.log:
            print(line)
        return

    def get_log_string(self):
        return "\n".join(self.log)
