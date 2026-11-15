# Misses at hit@1: semantic / hybrid + glossary

5 of 50 questions had no correct result at rank 1.

| category | count |
|---|---|
| hit below rank 1 | 5 |

`label:` categories are mistakes in questions.json (fix those first); the rest are retrieval failures.

## preprocessor-define (hit below rank 1)

**Question:** What does #define actually do to my code before it gets compiled?

*first correct result is at rank 3*

**Expected:** How the C++ Compiler Works (`3tIqpEmWMLI`) @ 8:49

> include end brace and then right click on our hello world project and hit properties under cc++ and then pre-processor I'm going to set the pre-process to a file to yes make sure that you're editing your current configuration and platform so that these ...

**Retrieved:**

1. How the C++ Compiler Works (`3tIqpEmWMLI`) @ 9:34: "instead what I'm going to do is actually just use something called if the if pre-processor statement can let us include or exclude code based on a given condition so over here I'm just going to ..."
2. How the C++ Compiler Works (`3tIqpEmWMLI`) @ 5:42: "result variable nice and simple that's it let's hit control 7 to build that file you can see over here that it's built it successfully I'm actually going to just resize Visual Studio a little bit ..."
3. How the C++ Compiler Works (`3tIqpEmWMLI`) @ 7:57: "include end brace and then right click on our hello world project and hit properties under cc++ and then pre-processor I'm going to set the pre-process to a file to yes make sure that you're ..."

## duplicate-symbols (hit below rank 1)

**Question:** My project won't link because a function is "already defined" even though I only wrote it once. Why does this happen and how do I fix it?

*first correct result is at rank 3*

**Expected:** How the C++ Linker Works (`H4s55GgAg0I`) @ 9:44

> errors great okay so the other type of linking error that's pretty common is when we have duplicate symbols so in other words we have functions or variables which have the same name and the same signature so two identically named functions which have the ...

**Expected:** How the C++ Linker Works (`H4s55GgAg0I`) @ 13:00

> like so into log. CVP and then also over here and now you can see that we of course do have two log functions so how do we fix this well we've got a few options here if we undo all of this so that we're including log again we could mark this log function ...

**Retrieved:**

1. How the C++ Linker Works (`H4s55GgAg0I`) @ 10:46: "only have one definition of log in this file so it's not going to give us a compiling error if I hit contrl F7 just to compile this you can see it's totally fine but now if I build we get a ..."
2. How the C++ Linker Works (`H4s55GgAg0I`) @ 7:27: "linking error and you might be like what but why is that happening I'm not calling multiply anywhere why is it complaining about a linking error shouldn't it have just removed the function ..."
3. How the C++ Linker Works (`H4s55GgAg0I`) @ 12:52: "like so into log. CVP and then also over here and now you can see that we of course do have two log functions so how do we fix this well we've got a few options here if we undo all of this so that ..."

## function-call-cost (hit below rank 1)

**Question:** Is there a performance cost to calling a function? What actually happens when I call one?

*first correct result is at rank 4*

**Expected:** Functions in C++ (`V9zuox47zr0`) @ 6:07

> rid of all of this code and this is what we end up with a nice clean and easy to read program if I launch my program you can see that we get the correct values here so this is a pretty simple example but I think it's effective in demonstrating that ...

**Retrieved:**

1. How the C++ Compiler Works (`3tIqpEmWMLI`) @ 16:13: "your function we'll talk more about this in the linking video but essentially when we have multiple objs and our functions are defined in multiple objs it's going to be the linker's job to link ..."
2. Functions in C++ (`V9zuox47zr0`) @ 3:19: "copy and paste it a few times I'll call this something like result 2 result 3 we'll do 8 * 5 90 * 45 and if I run my program and oh hang on a minute why am I getting the same value everywhere oh ..."
3. Functions in C++ (`V9zuox47zr0`) @ 2:30: "B and we returned the product of those two integers so how do we call this function well calling a function is pretty simple let's go ahead and try and print the result of a multiplication I'm ..."

## include-guards (hit below rank 1)

**Question:** I've seen headers that use #ifndef and #define instead of #pragma once. How does that work, and are they the same thing?

*first correct result is at rank 2*

**Expected:** C++ Header Files (`9RJTQmK0YPI`) @ 9:48

> that log is already been included and it does not include it twice now there is one other way that we could add a header guard and I actually kind of like this four teaching purposes because it makes a little bit more sense than some kind of pragma once ...

**Expected:** C++ Header Files (`9RJTQmK0YPI`) @ 11:07

> really easy way to demonstrate this is just to copy and paste this entire file into our log CPP file and then if we take a look at this I'll also comment out log. H and common. H so you can see over here that the first time this is all fine it includes the ...

**Retrieved:**

1. C++ Header Files (`9RJTQmK0YPI`) @ 7:02: "program we can see that we managed to initialize our log and then log hello world to the console fantastic all right brilliant so let's go back to that header file and take a look at what that ..."
2. C++ Header Files (`9RJTQmK0YPI`) @ 9:44: "that log is already been included and it does not include it twice now there is one other way that we could add a header guard and I actually kind of like this four teaching purposes because it ..."
3. C++ Header Files (`9RJTQmK0YPI`) @ 11:35: "styles you may encounter this if and Def header guard so just be wary of what it is but again I would never write if and F I would always use pragma ones the last thing I want to show you is with ..."

## ref-no-reassign (hit below rank 1)

**Question:** Why can't I make a reference point to a different variable after I've created it?

*first correct result is at rank 2*

**Expected:** REFERENCES in C++ (`IzoFn3dfsPA`) @ 7:28

> in a and since it is being passed by reference we've basically Rewritten the code to do exactly the same thing internally when it gets compiled it will be exactly the same as what we've written before however this time it just looks nicer in our source ...

**Expected:** REFERENCES in C++ (`IzoFn3dfsPA`) @ 8:25

> references is that once you declare a reference you cannot change what it references what I mean by that is that suppose that I had two integers I Had A and B and B was set to eight a was set to five and then what I wanted to do is declare a reference that ...

**Retrieved:**

1. REFERENCES in C++ (`IzoFn3dfsPA`) @ 1:52: "code let's say that I make a variable I'll call it a and I'll set it equal to 5 it's just going to be an integer if I want to create a reference to that variable I can do so by typing in the type ..."
2. REFERENCES in C++ (`IzoFn3dfsPA`) @ 7:30: "references is that once you declare a reference you cannot change what it references what I mean by that is that suppose that I had two integers I Had A and B and B was set to eight a was set to ..."
3. REFERENCES in C++ (`IzoFn3dfsPA`) @ 1:03: "reference an existing variable unlike a pointer where you could create a new pointer variable and then set it equal to null pointer or something like that set it equal to zero you can't do that ..."

