import pyglet
sound = pyglet.media.load('./test/test.wav', streaming=False)
sound.play()
pyglet.app.run()