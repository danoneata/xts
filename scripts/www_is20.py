import dominate

from dominate import tags

from toolz import partition


def load_results_1():
    return {
        "text": "Lorem ipsum",
        "video-path": None,
        "audio-orig": None,
        "audio-theirs": None,
        "audio-ours": None,
    }


doc = dominate.document(title="Speaker disentanglement in video-to-speech conversion")

with doc.head:
    # tags.link(rel="stylesheet", href="static/style.css")
    # tags.script(type="text/javascript", src="script.js")
    # Bootstrap
    tags.link(rel="stylesheet", href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css")
    tags.script(type="text/javascript", src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js")

with doc:
    with tags.body():
        with tags.div(cls="container"):

            tags.h1("Speaker disentanglement in video-to-speech conversion", cls="mt-5")
            tags.p("This web-page presents results for our Interspeech 2020 submission.")

            with tags.ul():
                with tags.li():
                    tags.a("video-to-speech", href="#video-to-speech.html")
                    tags.span("in which we show results for our baseline system;")
                with tags.li():
                    tags.a("speaker control", href="#speaker-control.html")
                    tags.span("in which we results for speaker control.")

            tags.h2("Video-to-speech", name="video-to-speech")
            tags.p("We show results for the seen scenario.")

            data = load_results_1()
            for row in partition(4, data):
                with tags.div(cls="row"):
                    for col in row:
                        with tags.div(cls="col-3"):
                            tags.span(str(col))

            tags.h2("Speaker control", name="speaker-control")
            tags.p("We show results for the unseen scenario.")

with open("output/www/is20/index.html", "w") as f:
    f.write(str(doc))
