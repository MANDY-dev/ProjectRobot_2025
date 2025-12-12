def on_forever():
    basic.show_leds("""
        . # # . .
        # . . # .
        # . . # .
        # . . # .
        . # # . .
        """)
    basic.pause(5000)
    basic.show_leds("""
        . . # . .
        . # # . .
        . . # . .
        . . # . .
        . # # # .
        """)
    basic.pause(5000)
    basic.show_leds("""
        # # # # .
        . . . # .
        . # # . .
        # . . . .
        # # # # .
        """)
    basic.pause(5000)
    basic.show_leds("""
        # # # # .
        . . . # .
        # # # # .
        . . . # .
        # # # # .
        """)
    basic.pause(5000)
basic.forever(on_forever)
