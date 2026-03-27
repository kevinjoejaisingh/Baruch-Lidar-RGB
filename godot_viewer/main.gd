extends Node3D

# ── Settings ────────────────────────────────────────────────────────────────
@export var move_speed: float = 5.0
@export var sprint_multiplier: float = 3.0
@export var mouse_sensitivity: float = 0.003
@export var smoothing: float = 12.0
@export_range(1.0, 20.0) var point_size: float = 2.0

# ── Nodes ───────────────────────────────────────────────────────────────────
var camera: Camera3D
var cloud_mesh_instance: MeshInstance3D
var cloud_material: ShaderMaterial
var label: Label
var file_dialog: FileDialog
var canvas: CanvasLayer

# ── State ───────────────────────────────────────────────────────────────────
var velocity: Vector3 = Vector3.ZERO
var mouse_captured: bool = false
var right_mouse_held: bool = false
var loading: bool = false
var cloud_loaded: bool = false


func _ready() -> void:
	# Camera
	camera = Camera3D.new()
	camera.fov = 75.0
	camera.far = 500.0
	add_child(camera)
	camera.make_current()

	# HUD
	canvas = CanvasLayer.new()
	add_child(canvas)
	label = Label.new()
	label.position = Vector2(20, 20)
	label.add_theme_font_size_override("font_size", 18)
	label.add_theme_color_override("font_color", Color.WHITE)
	label.add_theme_color_override("font_shadow_color", Color.BLACK)
	label.add_theme_constant_override("shadow_offset_x", 1)
	label.add_theme_constant_override("shadow_offset_y", 1)
	canvas.add_child(label)

	# Background
	var env := WorldEnvironment.new()
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color(0.08, 0.08, 0.12)
	environment.ambient_light_color = Color.WHITE
	environment.ambient_light_energy = 1.0
	env.environment = environment
	add_child(env)

	# File dialog
	file_dialog = FileDialog.new()
	file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	file_dialog.filters = PackedStringArray(["*.ply ; PLY Point Cloud", "*.bin ; Binary Point Cloud"])
	file_dialog.size = Vector2i(900, 600)
	file_dialog.title = "Open Point Cloud"
	file_dialog.file_selected.connect(_on_file_selected)
	canvas.add_child(file_dialog)

	# Check command-line args for a file path
	var args := OS.get_cmdline_args()
	var file_to_load := ""
	for arg: String in args:
		if arg.ends_with(".ply") or arg.ends_with(".bin"):
			file_to_load = arg
			break

	# Also check if bundled cloud.bin exists
	if file_to_load == "" and FileAccess.file_exists("res://cloud.bin"):
		file_to_load = "res://cloud.bin"

	if file_to_load != "":
		label.text = "Loading..."
		call_deferred("_load_file", file_to_load)
	else:
		_show_file_picker()


func _show_file_picker() -> void:
	label.text = "Select a .ply or .bin point cloud file  |  O = open file"
	file_dialog.current_dir = OS.get_environment("HOME")
	file_dialog.popup_centered()


func _on_file_selected(path: String) -> void:
	label.text = "Loading..."
	call_deferred("_load_file", path)


# ── File loading ────────────────────────────────────────────────────────────

func _load_file(path: String) -> void:
	if path.ends_with(".ply"):
		_load_ply(path)
	elif path.ends_with(".bin"):
		_load_bin(path)
	else:
		label.text = "ERROR: Unsupported format (use .ply or .bin)"


func _load_bin(path: String) -> void:
	loading = true
	var file := FileAccess.open(path, FileAccess.READ)
	if not file:
		label.text = "ERROR: Cannot open " + path
		loading = false
		return

	var n_points: int = file.get_32()
	label.text = "Loading %d points from .bin..." % n_points

	var pos_raw: PackedByteArray = file.get_buffer(n_points * 12)
	var col_raw: PackedByteArray = file.get_buffer(n_points * 12)
	file.close()

	var pos_f: PackedFloat32Array = pos_raw.to_float32_array()
	var col_f: PackedFloat32Array = col_raw.to_float32_array()

	var vertices := PackedVector3Array()
	vertices.resize(n_points)
	var colors := PackedColorArray()
	colors.resize(n_points)

	for i: int in range(n_points):
		var fi: int = i * 3
		vertices[i] = Vector3(pos_f[fi], pos_f[fi + 1], pos_f[fi + 2])
		colors[i] = Color(col_f[fi], col_f[fi + 1], col_f[fi + 2])

	_build_cloud(vertices, colors, path)


func _load_ply(path: String) -> void:
	loading = true
	label.text = "Loading PLY..."

	var file := FileAccess.open(path, FileAccess.READ)
	if not file:
		label.text = "ERROR: Cannot open " + path
		loading = false
		return

	# Parse PLY header
	var n_points: int = 0
	var has_color: bool = false
	var is_binary_le: bool = false
	var is_binary_be: bool = false
	var prop_names: PackedStringArray = []

	while true:
		var line: String = file.get_line().strip_edges()
		if line.begins_with("element vertex"):
			n_points = int(line.split(" ")[2])
		elif line.begins_with("property"):
			var parts: PackedStringArray = line.split(" ")
			if parts.size() >= 3:
				prop_names.append(parts[2])
				if parts[2] == "red" or parts[2] == "green" or parts[2] == "blue":
					has_color = true
		elif line == "format binary_little_endian 1.0":
			is_binary_le = true
		elif line == "format binary_big_endian 1.0":
			is_binary_be = true
		elif line == "end_header":
			break

	label.text = "Loading %d points from PLY..." % n_points

	if n_points == 0:
		label.text = "ERROR: No vertices in PLY"
		loading = false
		return

	# Build property layout — compute byte offset and size for each property
	var prop_types: PackedStringArray = []
	var prop_offsets: PackedInt32Array = []
	var prop_sizes: PackedInt32Array = []
	var byte_offset: int = 0

	# Re-parse header for types
	file.seek(0)
	while true:
		var hline: String = file.get_line().strip_edges()
		if hline.begins_with("property"):
			var hparts: PackedStringArray = hline.split(" ")
			if hparts.size() >= 3:
				var ptype: String = hparts[1]
				var psize: int = 4  # default float
				if ptype == "double" or ptype == "float64":
					psize = 8
				elif ptype == "float" or ptype == "float32":
					psize = 4
				elif ptype == "uchar" or ptype == "uint8":
					psize = 1
				elif ptype == "char" or ptype == "int8":
					psize = 1
				elif ptype == "short" or ptype == "int16" or ptype == "ushort" or ptype == "uint16":
					psize = 2
				elif ptype == "int" or ptype == "int32" or ptype == "uint" or ptype == "uint32":
					psize = 4
				prop_types.append(ptype)
				prop_offsets.append(byte_offset)
				prop_sizes.append(psize)
				byte_offset += psize
		elif hline == "end_header":
			break
	var stride: int = byte_offset

	# Find property indices
	var x_idx: int = prop_names.find("x")
	var y_idx: int = prop_names.find("y")
	var z_idx: int = prop_names.find("z")
	var r_idx: int = prop_names.find("red")
	var g_idx: int = prop_names.find("green")
	var b_idx: int = prop_names.find("blue")

	var vertices := PackedVector3Array()
	vertices.resize(n_points)
	var colors := PackedColorArray()
	colors.resize(n_points)

	if is_binary_le:
		var remaining: PackedByteArray = file.get_buffer(n_points * stride)
		print("PLY binary: %d points, stride=%d bytes, total=%d bytes" % [n_points, stride, remaining.size()])

		var x_off: int = prop_offsets[x_idx]
		var y_off: int = prop_offsets[y_idx]
		var z_off: int = prop_offsets[z_idx]
		var x_is_double: bool = prop_types[x_idx] in ["double", "float64"]

		var r_off: int = prop_offsets[r_idx] if r_idx >= 0 else -1
		var g_off: int = prop_offsets[g_idx] if g_idx >= 0 else -1
		var b_off: int = prop_offsets[b_idx] if b_idx >= 0 else -1
		var color_is_uchar: bool = r_idx >= 0 and prop_types[r_idx] in ["uchar", "uint8"]

		for i: int in range(n_points):
			var off: int = i * stride

			var px: float
			var py: float
			var pz: float
			if x_is_double:
				px = remaining.decode_double(off + x_off)
				py = remaining.decode_double(off + y_off)
				pz = remaining.decode_double(off + z_off)
			else:
				px = remaining.decode_float(off + x_off)
				py = remaining.decode_float(off + y_off)
				pz = remaining.decode_float(off + z_off)

			# Coordinate conversion: swap Y/Z for Godot (Y-up)
			vertices[i] = Vector3(px, pz, -py)

			if r_off >= 0:
				if color_is_uchar:
					var r: float = float(remaining.decode_u8(off + r_off)) / 255.0
					var g: float = float(remaining.decode_u8(off + g_off)) / 255.0
					var b: float = float(remaining.decode_u8(off + b_off)) / 255.0
					colors[i] = Color(r, g, b)
				else:
					colors[i] = Color(
						remaining.decode_float(off + r_off),
						remaining.decode_float(off + g_off),
						remaining.decode_float(off + b_off))
			else:
				colors[i] = Color(0.7, 0.7, 0.7)
	else:
		# ASCII PLY
		for i: int in range(n_points):
			var line: String = file.get_line().strip_edges()
			var parts: PackedStringArray = line.split(" ")
			if parts.size() < 3:
				continue

			var px: float = float(parts[x_idx]) if x_idx >= 0 else 0.0
			var py: float = float(parts[y_idx]) if y_idx >= 0 else 0.0
			var pz: float = float(parts[z_idx]) if z_idx >= 0 else 0.0
			vertices[i] = Vector3(px, pz, -py)

			if has_color and r_idx >= 0 and parts.size() > b_idx:
				var r: float = float(parts[r_idx]) / 255.0
				var g: float = float(parts[g_idx]) / 255.0
				var b: float = float(parts[b_idx]) / 255.0
				colors[i] = Color(r, g, b)
			else:
				colors[i] = Color(0.7, 0.7, 0.7)

	file.close()
	_build_cloud(vertices, colors, path)


func _build_cloud(vertices: PackedVector3Array, colors: PackedColorArray, path: String) -> void:
	# Remove old cloud if reloading
	if cloud_mesh_instance:
		cloud_mesh_instance.queue_free()
		cloud_mesh_instance = null

	label.text = "Building mesh (%d points)..." % vertices.size()

	var arrays: Array = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_COLOR] = colors

	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_POINTS, arrays)

	# Shader material
	var shader := load("res://point_cloud.gdshader") as Shader
	cloud_material = ShaderMaterial.new()
	cloud_material.shader = shader
	cloud_material.set_shader_parameter("point_size", point_size)
	mesh.surface_set_material(0, cloud_material)

	cloud_mesh_instance = MeshInstance3D.new()
	cloud_mesh_instance.mesh = mesh
	add_child(cloud_mesh_instance)

	# Position camera
	var aabb: AABB = mesh.get_aabb()
	var center: Vector3 = aabb.get_center()
	camera.position = center + Vector3(0, 2, 5)
	camera.look_at(center)

	loading = false
	cloud_loaded = true
	var fname: String = path.get_file()
	_show_status("Loaded %s (%d points)" % [fname, vertices.size()])
	print("Loaded %d points from %s" % [vertices.size(), path])


# ── Input ───────────────────────────────────────────────────────────────────

func _input(event: InputEvent) -> void:
	# Mouse look — works when captured OR when right-click dragging
	if event is InputEventMouseMotion and not loading:
		if mouse_captured or right_mouse_held:
			camera.rotate_y(-event.relative.x * mouse_sensitivity)
			camera.rotate_object_local(Vector3.RIGHT, -event.relative.y * mouse_sensitivity)
			var rot: Vector3 = camera.rotation
			rot.x = clampf(rot.x, -PI * 0.45, PI * 0.45)
			camera.rotation = rot

	# Right-click drag to look
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_RIGHT:
			right_mouse_held = event.pressed
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed and not mouse_captured and cloud_loaded:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
			mouse_captured = true

	# Touchpad pan gesture
	if event is InputEventPanGesture and not loading:
		camera.rotate_y(-event.delta.x * mouse_sensitivity * 10.0)
		camera.rotate_object_local(Vector3.RIGHT, -event.delta.y * mouse_sensitivity * 10.0)
		var rot: Vector3 = camera.rotation
		rot.x = clampf(rot.x, -PI * 0.45, PI * 0.45)
		camera.rotation = rot

	# Key actions
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_ESCAPE:
				if mouse_captured:
					Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
					mouse_captured = false
			KEY_O:
				if not mouse_captured:
					_show_file_picker()
			KEY_1: _set_point_size(1.0)
			KEY_2: _set_point_size(2.0)
			KEY_3: _set_point_size(3.0)
			KEY_4: _set_point_size(4.0)
			KEY_5: _set_point_size(5.0)
			KEY_6: _set_point_size(8.0)
			KEY_7: _set_point_size(12.0)
			KEY_EQUAL: move_speed *= 1.5; _show_status("Speed: %.1f" % move_speed)
			KEY_MINUS: move_speed /= 1.5; _show_status("Speed: %.1f" % move_speed)


func _set_point_size(size: float) -> void:
	point_size = size
	if cloud_material:
		cloud_material.set_shader_parameter("point_size", size)
	_show_status("Point size: %.0f" % size)


func _show_status(text: String) -> void:
	label.text = text
	get_tree().create_timer(2.0).timeout.connect(func(): label.text = "")


# ── Movement ────────────────────────────────────────────────────────────────

func _physics_process(delta: float) -> void:
	if loading or not cloud_loaded:
		return

	var input_dir := Vector3.ZERO
	if Input.is_key_pressed(KEY_W): input_dir.z -= 1.0
	if Input.is_key_pressed(KEY_S): input_dir.z += 1.0
	if Input.is_key_pressed(KEY_A): input_dir.x -= 1.0
	if Input.is_key_pressed(KEY_D): input_dir.x += 1.0
	if Input.is_key_pressed(KEY_SPACE): input_dir.y += 1.0
	if Input.is_key_pressed(KEY_C) or Input.is_key_pressed(KEY_CTRL): input_dir.y -= 1.0

	if input_dir.length() > 0:
		input_dir = input_dir.normalized()

	var direction: Vector3 = camera.global_basis * input_dir
	var speed: float = move_speed
	if Input.is_key_pressed(KEY_SHIFT):
		speed *= sprint_multiplier

	var target_vel: Vector3 = direction * speed
	velocity = velocity.lerp(target_vel, clampf(smoothing * delta, 0.0, 1.0))
	camera.global_position += velocity * delta
