" (c) 1990, 1991, 1992, 1993, 1994, 1995 Copyright (c) University of Washington

  All rights reserved. Use of this software is permitted for non-commercial
  research purposes, and it may be copied only for that use.  All copies must
  include this copyright message.  This software is made available AS IS, and
  neither the authors nor the University of Washington make any warranty about
  the software or its performance.

  When you first acquire this software please send mail to
  bug-snlp@cs.washington.edu; the same address should be used for problems."

(asdf:defsystem "ucpop40"
  :description "UCPOP 4.0 Planning System"
  :version "4.0"
  :author "University of Washington"
  :license "Non-commercial research use only"
  :depends-on ()
  :serial t
  :components
  ((:file "variable")
   (:file "choose")
   (:file "rules")
   (:file "scr" :depends-on ("choose" "rules"))
   (:file "package" :depends-on ("variable"))
   (:file "struct" :depends-on ("package"))
   (:file "plan-utils" :depends-on ("package" "struct" "variable"))
   (:file "ucpop" :depends-on ("package" "struct"))
   (:file "zlifo" :depends-on ("package" "struct"))
   (:file "safety" :depends-on ("package"))
   (:file "interface" :depends-on ("scr" "plan-utils" "struct"))
   (:file "controllers" :depends-on ("interface"))
   (:module "domains"
    :pathname "domains/"
    :depends-on ("interface")
    :components
    ((:file "domains")
     (:file "truckworld")
     (:file "safety-domain")))))

(defun compile-ucpop () (asdf:compile-system "ucpop40" :force t))
(defun load-ucpop () (asdf:load-system "ucpop40" :force t))
