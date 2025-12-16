#!/usr/bin/env python3
import os, math, random, time
import pygame
from collections import deque

WIDTH, HEIGHT = 1600, 900
FPS_TARGET = 60

# Modes
RAINBOW, HUGE = 0, 1
MODE_NAMES = ["Rainbow", "HUGE"]

def clamp(x, a, b): return a if x < a else b if x > b else x
def randf(a,b): return random.uniform(a,b)

def hsv_to_rgb(h):  # h 0..6
    c = 255
    x = int(255 * (1 - abs((h % 2) - 1)))
    if 0 <= h < 1: return (c, x, 0)
    if 1 <= h < 2: return (x, c, 0)
    if 2 <= h < 3: return (0, c, x)
    if 3 <= h < 4: return (0, x, c)
    if 4 <= h < 5: return (x, 0, c)
    return (c, 0, x)

# ---------- layered sphericalized skybox ----------

class SkyLayer:
    """rainbow!!"""
    def __init__(self, tiles=16, radius=1.0, intensity=1.0, hue_shift=0.0):
        self.tiles = tiles
        self.radius = radius
        self.intensity = intensity
        self.hue_shift = hue_shift
        self.colors = self.make_colors()
        self.row_shift = [0]*6
        self.col_shift = [0]*6
        self.seeds = [random.random()*1000 for _ in range(6)]
        
        self._cache_pts = None
        self._cache_frame = -999

    def make_colors(self):
        N = self.tiles
        faces = [[[(0,0,0) for _ in range(N)] for _ in range(N)] for _ in range(6)]
        for f in range(6):
            for y in range(N):
                for x in range(N):
                    h = (randf(0,6) + self.hue_shift) % 6
                    col = hsv_to_rgb(h)
                   
                    col = tuple(clamp(ch + random.randint(-12,12), 0, 255) for ch in col)
                    faces[f][y][x] = col
       
        neighbors = {
            0: {0:(4,0), 1:(2,3), 2:(5,0), 3:(3,1)},
            1: {0:(4,2), 1:(3,3), 2:(5,2), 3:(2,1)},
            2: {0:(4,3), 1:(0,3), 2:(5,3), 3:(1,1)},
            3: {0:(4,1), 1:(1,3), 2:(5,1), 3:(0,1)},
            4: {0:(1,2), 1:(0,0), 2:(2,0), 3:(3,0)},
            5: {0:(2,2), 1:(0,2), 2:(1,0), 3:(3,2)},
        }
        N1 = N-1
        for f in range(6):
            tf, te = neighbors[f][0]
            for x in range(N):
                faces[f][0][x] = faces[tf][N1 if te==2 else 0][x if te in (0,2) else (0 if te==3 else N1)]
            rf, re = neighbors[f][1]
            for y in range(N):
                faces[f][y][N1] = faces[rf][y if re in (1,3) else (0 if re==0 else N1)][N1 if re==3 else 0]
            bf, be = neighbors[f][2]
            for x in range(N):
                faces[f][N1][x] = faces[bf][0 if be==0 else N1][x if be in (0,2) else (0 if be==3 else N1)]
            lf, le = neighbors[f][3]
            for y in range(N):
                faces[f][y][0] = faces[lf][y if le in (1,3) else (0 if le==0 else N1)][0 if le==1 else N1]
        return faces

    def step(self, t):
        base = 0.12 * self.intensity
        for f in range(6):
            if random.random() < base:
                self.row_shift[f] = (self.row_shift[f] + random.choice([-1,0,1])) % self.tiles
            if random.random() < base:
                self.col_shift[f] = (self.col_shift[f] + random.choice([-1,0,1])) % self.tiles
       
        if random.random() < 0.03:
            self.hue_shift = (self.hue_shift + randf(-0.02, 0.02)) % 6
            self.colors = self.make_colors()


def face_to_sphere(radius, f, u, v):
    s = 1.0
    if f == 0: x,y,z = ( s,    v, -u)
    elif f == 1: x,y,z = (-s,   v,  u)
    elif f == 2: x,y,z = ( u,   s, -v)
    elif f == 3: x,y,z = ( u,  -s,  v)
    elif f == 4: x,y,z = ( u,   v,  s)
    else:        x,y,z = (-u,   v, -s)
    L = (x*x + y*y + z*z) ** 0.5 or 1.0
    return (radius * x / L, radius * y / L, radius * z / L)

class Camera:
    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.pos = [0.0, 0.0, 0.0]
        self.speed = 1.0

    def forward_flat(self):
        return (math.sin(self.yaw), 0, math.cos(self.yaw))
    def right(self):
        return (math.cos(self.yaw), 0, -math.sin(self.yaw))

    def apply(self, v):
        x,y,z = v[0]-self.pos[0], v[1]-self.pos[1], v[2]-self.pos[2]
        cy, sy = math.cos(-self.yaw), math.sin(-self.yaw)
        x, z = cy*x - sy*z, sy*x + cy*z
        cp, sp = math.cos(-self.pitch), math.sin(-self.pitch)
        y, z = cp*y - sp*z, sp*y + cp*z
        return (x, y, z)

def project(v, cx, cy, f):
    x,y,z = v
    z = (z + 3.2)
    if z <= 0.1: z = 0.1
    s = f / z
    return (cx + x*s, cy - y*s)

def draw_layer(surface, cam, layer: SkyLayer, t, precision=False, warp_cache_every=3):
   
    if layer._cache_pts is None:
        layer._cache_pts = {f: [[None]*layer.tiles for _ in range(layer.tiles)] for f in range(6)}

    N = layer.tiles
    wob = 0.008 * layer.intensity  
    melt = 0.02 if precision else 0.0

    w,h = surface.get_size()
    cx, cy = w/2, h/2
    f = 780 * (w / WIDTH) 

    need_recalc = (pygame.time.get_ticks() // (1000//FPS_TARGET)) % warp_cache_every == 0

    for face in (5,1,0,3,2,4):
        seed = layer.seeds[face]
        for iy in range(N):
            for ix in range(N):
                jx = (ix + layer.col_shift[face]) % N
                jy = (iy + layer.row_shift[face]) % N
                col = layer.colors[face][jy][jx]
               
                jitter = int(30 * layer.intensity)
                col = (clamp(col[0]+random.randint(-jitter,jitter),0,255),
                       clamp(col[1]+random.randint(-jitter,jitter),0,255),
                       clamp(col[2]+random.randint(-jitter,jitter),0,255))

                if need_recalc or layer._cache_pts[face][iy][ix] is None:
                    u0 = -1 + 2 * (ix / N); v0 = -1 + 2 * (iy / N)
                    u1 = -1 + 2 * ((ix+1) / N); v1 = -1 + 2 * ((iy+1) / N)

                    def warp(u, v, kadd):
                        k = seed + ix*0.31 + iy*0.71 + kadd
                        u += wob * math.sin(t*2.5 + k) + wob*0.5*math.sin((u+v+t)*4.8 + k*1.9)
                        v += wob * math.cos(t*2.0 + k*0.6) + wob*0.5*math.sin((u-t)*5.2 + k*2.6)
                        if melt:
                            u += (random.random()-0.5) * melt
                            v += (random.random()-0.5) * melt
                        return u, v

                    du0, dv0   = warp(u0, v0, 0)
                    du1, dv0b  = warp(u1, v0, 1)
                    du1b, dv1  = warp(u1, v1, 2)
                    du0b, dv1b = warp(u0, v1, 3)

                    corners = [
                        face_to_sphere(layer.radius, face, du0, dv0),
                        face_to_sphere(layer.radius, face, du1, dv0b),
                        face_to_sphere(layer.radius, face, du1b, dv1),
                        face_to_sphere(layer.radius, face, du0b, dv1b),
                    ]
                    pts2d = [project(cam.apply(p), cx, cy, f) for p in corners]
                    layer._cache_pts[face][iy][ix] = pts2d
                else:
                    pts2d = layer._cache_pts[face][iy][ix]

                pygame.draw.polygon(surface, col, pts2d, 0)

def main():
    pygame.init()
    pygame.display.set_caption("colorbox")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

  
    base_radii = [1.4, 6.0]
    base_tiles = [16, 14]  

    layers = [SkyLayer(base_tiles[i], base_radii[i], intensity=1.0 - i*0.2, hue_shift=i*0.9)
              for i in range(len(base_radii))]

    cam = Camera()
    mouse_locked = False
    invert_pitch = False
    precision = False
    mode = RAINBOW

  
    scale = 0.85  
    rsurf = pygame.Surface((int(WIDTH*scale), int(HEIGHT*scale))).convert()

    
    fps_hist = deque(maxlen=30)

    pygame.mouse.set_visible(True)

    running = True
    while running:
        dt = clock.tick(FPS_TARGET) / 1000.0
        t = pygame.time.get_ticks() / 1000.0
        wheel_move = 0.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: running = False
                elif e.key == pygame.K_l:
                    mouse_locked = not mouse_locked
                    pygame.mouse.set_visible(not mouse_locked)
                    pygame.event.set_grab(mouse_locked)
                elif e.key == pygame.K_i:
                    invert_pitch = not invert_pitch
                elif e.key == pygame.K_f:
                    for lay in layers: lay.intensity = clamp(lay.intensity + 0.3, 0.5, 4.0)
                elif e.key == pygame.K_p:
                    precision = not precision
                elif e.key == pygame.K_m:
                    if mode == RAINBOW:
                        mode = HUGE
                        for i, lay in enumerate(layers):
                            lay.radius = base_radii[i] * 12.0
                            lay.tiles = max(10, int(base_tiles[i] * 0.75))  
                            lay.colors = lay.make_colors()
                    else:
                        mode = RAINBOW
                        for i, lay in enumerate(layers):
                            lay.radius = base_radii[i]
                            lay.tiles = base_tiles[i]
                            lay.colors = lay.make_colors()
                elif e.key == pygame.K_z:
                    cam.pos = [0.0, 0.0, 0.0]; cam.yaw = 0.0; cam.pitch = 0.0
            elif e.type == pygame.MOUSEWHEEL:
                wheel_move += e.y

        
        if mouse_locked:
            mx, my = pygame.mouse.get_rel()
            cam.yaw += mx * 0.0030
            if invert_pitch:
                cam.pitch = clamp(cam.pitch + my * 0.0030, -1.2, 1.2)
            else:
                cam.pitch = clamp(cam.pitch - my * 0.0030, -1.2, 1.2)
        else:
            pygame.mouse.get_rel()

        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:  cam.yaw -= 0.035
        if keys[pygame.K_RIGHT]: cam.yaw += 0.035
        if keys[pygame.K_UP]:    cam.pitch = clamp(cam.pitch - 0.035, -1.2, 1.2)
        if keys[pygame.K_DOWN]:  cam.pitch = clamp(cam.pitch + 0.035, -1.2, 1.2)

       
        boost = 2.2 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0
        spd = cam.speed * dt * boost
        fx, _, fz = cam.forward_flat()
        rx, _, rz = cam.right()
        if keys[pygame.K_w]: cam.pos[0] += fx*spd; cam.pos[2] += fz*spd
        if keys[pygame.K_s]: cam.pos[0] -= fx*spd; cam.pos[2] -= fz*spd
        if keys[pygame.K_a]: cam.pos[0] -= rx*spd; cam.pos[2] -= rz*spd
        if keys[pygame.K_d]: cam.pos[0] += rx*spd; cam.pos[2] += rz*spd
        if keys[pygame.K_q]: cam.pos[1] -= spd
        if keys[pygame.K_e]: cam.pos[1] += spd

        
        if wheel_move != 0.0:
            mx, my = pygame.mouse.get_pos()
            f = 760.0 * scale
            rx_cam = (mx - WIDTH/2) / (760.0)
            ry_cam = -(my - HEIGHT/2) / (760.0)
            rz_cam = 1.0
            cy, sy = math.cos(cam.yaw), math.sin(cam.yaw)
            cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
            x1, y1, z1 = rx_cam, ry_cam*cp + rz_cam*sp, -ry_cam*sp + rz_cam*cp
            dx, dy, dz = x1*cy - z1*sy, y1, x1*sy + z1*cy
            L = (dx*dx + dy*dy + dz*dz) ** 0.5 or 1.0
            dx, dy, dz = dx/L, dy/L, dz/L
            wheel_speed = 1.4 * wheel_move * (1.0 + 0.5*boost)
            cam.pos[0] += dx * wheel_speed * dt * 60.0
            cam.pos[1] += dy * wheel_speed * dt * 60.0
            cam.pos[2] += dz * wheel_speed * dt * 60.0

        
        for lay in layers: lay.step(t)

        
        iw, ih = rsurf.get_size()
        rsurf.fill((0,0,0))
        for lay in layers[::-1]:
            draw_layer(rsurf, cam, lay, t, precision, warp_cache_every=3)

        
        pygame.transform.smoothscale(rsurf, (WIDTH, HEIGHT), screen)

        
        fps = clock.get_fps()
        fps_hist.append(fps if fps > 0 else FPS_TARGET)
        avg = sum(fps_hist)/len(fps_hist)
        if len(fps_hist) == fps_hist.maxlen:
            if avg < 30 and scale > 0.6:
                scale = max(0.6, scale - 0.1)
                rsurf = pygame.Surface((int(WIDTH*scale), int(HEIGHT*scale))).convert()
            elif avg > 58 and scale < 1.0:
                scale = min(1.0, scale + 0.1)
                rsurf = pygame.Surface((int(WIDTH*scale), int(HEIGHT*scale))).convert()

        
        font = pygame.font.SysFont("consolas,monospace", 16)
        hud = f"[Mode {MODE_NAMES[mode]}] [Scale {scale:.2f}] [FPS {fps:.1f}]"
        help1 = "L lock | Wheel move-to-cursor | WASD+QE | Shift | F glitch | P precision | M HUGE | I invert | Z reset | ESC"
        screen.blit(font.render(hud, True, (240,240,245)), (10, 8))
        screen.blit(font.render(help1, True, (185,185,195)), (10, 28))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
