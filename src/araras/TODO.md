Implemented zellij pane launching for monitor.
But, maybe close the panel if the code crashes? Or keep it for logging?

Make scheduled restarts smart, using memory info, like the watchdog does, to avoid restarting when not necessary. Something simple, such as threasholds and pooling from time to time to check if the memory usage is above a certain limit before restarting.