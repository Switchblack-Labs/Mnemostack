# Mnemostack, where this is going

## The idea in one breath

Coding agents go blind the moment your work crosses from one repo into another. If
one service leans on a shared library next door, the agent has no idea that library
exists. You end up cloning the other repos and pasting code in by hand. We want to
get rid of that. One graph, running on your machine, that stretches across the few
repos you actually work in at once, so the agent can follow a thread from one into
another by itself.

This is not about indexing a whole company's code. That just turns us into a weaker
version of tools that already exist. It's about the small cluster of repos one person
has open at the same time. A service, the library it uses, the thing that calls it.
A few repos that are already on the machine.

## Why it beats the single repo version

Inside one repo our advantage is thin. The agent already has everything, so we're
just saving it a little legwork. Not a reason to exist.

Across repos the gap is real. The other repo isn't slow to find, it's simply not
there for the agent to see at all. So we're not competing with search, we're handing
over context that had no way of showing up otherwise. That's the whole point.

## What's actually next

The work has a natural order because each part leans on the one before it.

- Connecting code across files properly. Right now we only link things that sit in
  the same file, and that one limitation is what holds back everything else.
- Following imports to the real files instead of guessing at them, since that same
  skill is what lets us hop between repos later.
- Letting the graph know which repo each piece of code came from, so it can hold
  several repos at once.
- Reading the dependency files every repo already has, which give us the cleanest
  possible link between repos because the repo itself is telling us.
- Being honest, per link, about how sure we are, so a shaky guess never gets dressed
  up as a fact.

The thing we demo is that fourth point. Two real repos where one depends on the
other, ask about something in the first, watch the answer reach into the second.
That's the moment it clicks for anyone watching.

## Outbound is easy, inbound is the hard part

There are two directions and they're not the same difficulty.

Following your own dependencies outward basically just works. When a dependency
traces out, either the other repo is on disk, so we resolve the path, index it, and
keep going. Or it isn't, and we just pop a note saying "this traces to repo B, not
indexed," which is useful on its own because now the agent knows where the boundary
is. And the big one: installed packages are already on disk. The source in
site-packages or node_modules is the dependency's actual code, sitting right there.
So most of "the other repo" is present without cloning anything.

Going the other way is where it's limited. "Who calls this" or "who depends on me"
needs us to know about repos that aren't your dependencies and aren't on your disk.
You can't point at a repo you've never seen. That needs something to have already
indexed the other side, which is a shared-index, whole-org problem.

So we do outbound now because it's real and it works, and v1 is entirely outbound:
following your own dependencies out through the manifest files. Inbound waits until a
shared index actually makes sense.

## The honest limit we have to own

What we built reads code without running it. That's fast, safe, and great for
anything written in a straightforward way. But it goes blind on the clever stuff, and
especially on anything that talks over the network. One service calling another isn't
something we can see in the code, it's just a line with a web address in it.

This matters because the links between repos are often that kind. So we don't promise
a complete map. We promise a precise map of what's solid, and we're upfront that the
network side needs a different approach. There's a sensible way to get there later by
reading the contract files that describe those connections (an API spec, the route
definitions), but that's a problem for after the core works, not now.

## Languages

We go deep on a few and shallow on the rest, on purpose. A handful of languages get
the full treatment with real connections between functions. Everything else still
gets searched and chunked, just without the deep links. Promising the deep version
for every language is a trap, because the clever runtime behavior looks different in
each one and there's no shortcut around that.

## The bigger calls we owe ourselves

A few decisions shape the whole thing and we should make them on purpose rather than
drift into them.

- How much we trust the graph. It's precise on what it can see and quiet on what it
  can't, so when it isn't sure the agent should fall back to plain search instead of
  leaning on a half answer.
- Whether we ever run code to learn more, or stay safely on the outside reading it.
  Staying outside is simpler and we lean that way for now.
- Build our own understanding of each language or borrow existing tools that already
  do it. Building gives us control, borrowing gives us speed.
- What's actually the headline. The cross-repo graph or the session memory. We've
  been treating the graph as the real product, and that feels right.
