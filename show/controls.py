"""All operator shortcuts in one table. The operator panel builds its buttons and its
help page from this table, so keys, buttons and help always match."""
import pygame

# (key code, key name, help text)
SHORTCUTS = [
    (pygame.K_SPACE, "SPACE", "Attract mode: start a session. In a stage: the stage action "
                              "(stage 1: take the photo; stage 2 game and stage 3: start a round)."),
    (pygame.K_RIGHT, "RIGHT", "Next stage. After the last stage: back to attract mode."),
    (pygame.K_LEFT, "LEFT", "Previous stage."),
    (pygame.K_ESCAPE, "ESC", "Back to attract mode."),
    (pygame.K_p, "P", "Jump to the simulated attack (backup when the evasion game fails)."),
    (pygame.K_BACKSPACE, "BACKSPACE", "Stage 1: drop the photo and go back to the live video."),
    (pygame.K_1, "1 - 4", "Stage 2 games: 1 'Fool the AI', 2 'What does the AI think?' (SPACE: freeze, reveal, next), 3 'Scavenger hunt', 4 'Confusion meter'."),
    (pygame.K_0, "0", "Stage 2: back to free detection."),
    (pygame.K_i, "I", "Show / hide the 'how to play' card of the current game (stage 2 game, stage 3)."),
    (pygame.K_t, "T", "Fool the AI: choose another target (between rounds)."),
    (pygame.K_n, "N", "After a round: type the team name, ENTER saves, ESC cancels."),
    (pygame.K_r, "Shift+R", "Empty today's leaderboards (press twice within 3 s)."),
    (pygame.K_l, "L", "Switch the screen language: Dutch / English."),
    (pygame.K_c, "C", "Switch to the next camera."),
    (pygame.K_f, "F", "Fullscreen on / off for the show window."),
    (pygame.K_d, "D", "Debug overlay on the show screen (fps, boxes, player values)."),
    (pygame.K_LEFTBRACKET, "[  ]", "Debug on: player face size limit down / up."),
    (pygame.K_MINUS, "-  =", "Debug on: 'face visible' limit down / up."),
    (pygame.K_h, "H", "Help page in the operator panel on / off."),
    (pygame.K_q, "Q", "Quit the show: press twice within 3 s (also the red Quit button in the panel)."),
]
