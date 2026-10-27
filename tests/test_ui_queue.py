import threading

from agent.ui.queue import PromptQueue


def fill(q, *texts):
    for t in texts:
        q.enqueue(t)


def test_fifo_and_pause():
    q = PromptQueue()
    fill(q, "a", "b")
    q.pause()
    assert q.pop_next() is None and len(q) == 2
    q.resume()
    assert q.pop_next().text == "a" and q.pop_next().text == "b" and q.pop_next() is None


def test_remove_is_one_based():
    q = PromptQueue()
    fill(q, "a", "b", "c")
    assert q.remove(0) is None and q.remove(4) is None
    assert q.remove(2).text == "b"
    assert [i.text for i in q.items()] == ["a", "c"]
    assert q.remove(1).text == "a"


def test_move_clear_pull_back():
    q = PromptQueue()
    fill(q, "a", "b", "c")
    assert q.move(3, 1) and [i.text for i in q.items()] == ["c", "a", "b"]
    assert q.move(1, 99) and [i.text for i in q.items()] == ["a", "b", "c"]
    assert not q.move(9, 1)
    assert q.pull_back().text == "c" and len(q) == 2
    assert q.clear() == 2 and q.pull_back() is None


def test_ids_unique_kinds_notes():
    q = PromptQueue()
    a = q.enqueue("x"); b = q.enqueue("/clear", kind="command", note="after turn")
    assert a.id != b.id and b.kind == "command" and b.note == "after turn"


def test_lines():
    q = PromptQueue()
    assert q.lines() == []
    fill(q, "hello world " * 20)
    q.enqueue("/clear", "command")
    q.pause()
    ls = q.lines(30)
    assert ls[0].startswith("queued (2)") and "paused" in ls[0]
    assert all(len(l) <= 30 for l in ls) and ls[1].endswith("…") and "(cmd)" in ls[2]


def test_threads():
    q = PromptQueue()
    def prod():
        for i in range(200):
            q.enqueue(str(i))
    got = []
    def cons():
        for _ in range(2000):
            it = q.pop_next()
            if it:
                got.append(it.id)
    ts = [threading.Thread(target=prod) for _ in range(4)] + [threading.Thread(target=cons)]
    [t.start() for t in ts]; [t.join() for t in ts]
    while (it := q.pop_next()):
        got.append(it.id)
    assert len(got) == 800 and len(set(got)) == 800
