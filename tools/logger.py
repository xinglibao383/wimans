import os


class Logger:
    def __init__(self, save_path):
        self.save_path = os.path.join(save_path, 'log.txt')
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def record(self, logs):
        with open(self.save_path, "a", encoding="utf-8") as log_file:
            for log in logs:
                log_file.write(log + "\n")
                print(log)
