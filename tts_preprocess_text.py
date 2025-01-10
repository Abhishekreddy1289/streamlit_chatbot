import re
class TextCleaner:
    def __init__(self):
        # this is a static set of cleaning rules to be applied
        self.cleaning_rules = {
            " +" : " ",
            "^ +" : "",
            " +$" : "",
            "#" : "",
            "[.,;।!](\r\n)*" : ", ",
            "[.,;।!](\n)*" : ", ",
            "(\r\n)+" : ", ",
            "(\n)+" : ", ",
            "(\r)+" : ", ",
            """[|&’‘,।\\."]""": "",
            """[?;:)(!,]""": ", ", #added this because of give gap when this symbols in text
            "[/']" : "",
            "[-–]" : " ",
            "*":""
        }

    def clean(self, text):
        for key, replacement in self.cleaning_rules.items():
            text = re.sub(key, replacement, text)
        return text