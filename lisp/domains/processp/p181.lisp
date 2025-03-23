(load-start-state
 '(
	(is-a drill1 DRILL)

	(is-a spot-drill1 SPOT-DRILL)

	(is-a straight-fluted-drill1 STRAIGHT-FLUTED-DRILL)
	(diameter-of-drill-bit straight-fluted-drill1 1/32)

        (is-a toe-clamp1 TOE-CLAMP)

	(is-a brush1 BRUSH)

	(is-a soluble-oil SOLUBLE-OIL)
	(is-a mineral-oil MINERAL-OIL)
	
	(is-a part1 PART)
	(material-of part1 BRASS)
	(size-of part1 LENGTH 5)
	(size-of part1 WIDTH 3)
	(size-of part1 HEIGHT 2)
))

(load-goal

	'(exists (<part>) (is-a <part> PART) (and (has-hole <part> hole1 side1 1 1/32 1 1)
	      (is-available-part <part>))
)
)
