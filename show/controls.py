"""All operator shortcuts in one table. The operator panel builds its buttons and its
help page from this table, so keys, buttons and help always match."""
import pygame

# (key code, key name, help text)
SHORTCUTS = [
    (pygame.K_SPACE, "SPACE", "Attract mode: start a session. In a stage: the stage action "
                              "(stage 1: take the photo; stage 2 game and stage 3: start a round; during a round: stop it now "
                              "and show the score; during the countdown: cancel). Hit the number: measure the "
                              "maximum, start the countdown, next round."),
    (pygame.K_x, "X", "During a round (stage 2 game, stage 3): 10 seconds more. Press again for more."),
    (pygame.K_RIGHT, "RIGHT", "Next stage. After the last stage: back to attract mode (lesson: to the opening)."),
    (pygame.K_LEFT, "LEFT", "Previous stage."),
    (pygame.K_ESCAPE, "ESC", "Back to attract mode (lesson: the opening). With a leaderboard on screen: close it."),
    (pygame.K_p, "P", "Jump to the simulated attack (backup when the evasion game fails)."),
    (pygame.K_g, "G", "Switch between the pass-by show and the group lesson (also: python -m show --lesson)."),
    (pygame.K_BACKSPACE, "BACKSPACE", "Stage 1: drop the photo and go back to the live video. "
                                      "Hit the number: measure the maximum again."),
    (pygame.K_y, "Y", "Test mode on / off: a simulated group (bodies and faces), to test the lesson alone "
                      "(also: python -m show --test)."),
    (pygame.K_COMMA, ",  .", "Test mode on: hide one simulated face / show one again. "
                             "With Shift: remove one simulated person / add one."),
    (pygame.K_UP, "UP / DOWN", "Hit the number: target one higher / lower (before a round)."),
    (pygame.K_1, "0 - 4", "Stage 2 games: 1 Fool, 2 Guess (SPACE: freeze, reveal, next), 3 Hunt, 4 Confuse. 0: free detection."),
    (pygame.K_i, "I", "Show / hide the 'how to play' card of the current game (stage 2 game, stage 3)."),
    (pygame.K_e, "E", "Explanation. Stage 1 (after the photo): one real filter on the pixels. Stage 2: old AI -> "
                      "modern AI -> clues to boxes -> off. Stage 3: why the AI can miss a face. Stage 4: glasses."),
    (pygame.K_t, "T", "Fool: next target (Shift+T: back). Confuse: choose the next object; after the last one, all held objects count again."),
    (pygame.K_n, "N", "After a round: type the team name, ENTER saves, ESC cancels."),
    (pygame.K_r, "Shift+R", "Empty today's leaderboards (press twice within 3 s)."),
    (pygame.K_e, "Shift+E", "All explanations: simple (default, for everyone) / detailed (more numbers and steps)."),
    (pygame.K_PAGEDOWN, "PgDn / N", "Explanation open: show the next text step (a presentation clicker sends PgDn). "
                                    "A page opens with only its title and image; nothing moves on by itself."),
    (pygame.K_PAGEUP, "PgUp", "Explanation: one text step back."),
    (pygame.K_TAB, "TAB", "Explanations follow another face (the next one to the right). They never switch by themselves."),
    (pygame.K_m, "M", "Stage 3 face model: hard (YOLO) / easy (Haar)."),
    (pygame.K_v, "V", "Face scores on / off (stage 3: off = only in a round)."),
    (pygame.K_l, "L", "Switch the screen language: Dutch / English."),
    (pygame.K_c, "C", "Switch to the next camera."),
    (pygame.K_f, "F", "Fullscreen on / off for the show window."),
    (pygame.K_o, "O", "Overlays on / off: header, info bars, QR code."),
    (pygame.K_k, "K", "QR code on / off."),
    (pygame.K_d, "D", "Debug overlay: fps, boxes, player values, and why a person is not a player."),
    (pygame.K_LEFTBRACKET, "[  ]", "Debug on: player face size limit down / up."),
    (pygame.K_MINUS, "-  =", "Debug on: 'face seen' limit down / up (YOLO score or Haar hits)."),
    (pygame.K_h, "H", "Help page in the operator panel on / off."),
    (pygame.K_q, "Q", "Quit the show: press twice within 3 s (also the red Quit button in the panel)."),
]
