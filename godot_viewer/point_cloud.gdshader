shader_type spatial;
render_mode unshaded, cull_disabled;

uniform float point_size : hint_range(0.5, 30.0) = 2.0;

void vertex() {
	POINT_SIZE = point_size;
}

void fragment() {
	// Round points
	vec2 c = POINT_COORD - vec2(0.5);
	if (dot(c, c) > 0.25) discard;

	// PLY colors are sRGB, Godot expects linear input — convert
	ALBEDO = pow(COLOR.rgb, vec3(2.2));
}
