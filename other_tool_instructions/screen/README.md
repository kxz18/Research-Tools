# Common Commands for `screen`

start a session

```bash
screen -S session_name
```

start a session without attaching to it (daemon mode)

```bash
screen -dmS session_name
```

attach to an existing session

```bash
screen -r session_name
```

detach from a session

```bash
Ctrl + A + D
# or when you can still interact with the bash
screen -d
```

enter copy mode, where you can see the history logs

```bash
Ctrl + A + [
```

execute a command without attaching to a session

```bash
screen -r session_name -X stuff "cmd\n"
```

kill all sessions with the same keyword

```bash
screen -ls keyword | grep -E '\s+[0-9]+\.' | awk -F ' ' '{print $1}' | while read s; do screen -XS $s quit; done
```