#!/usr/bin/env python3
import os, sys, math, random, time
import pygame

WIDTH, HEIGHT = 1600, 900
FPS = 60

def clamp(x, a, b): return a if x < a else b if x > b else x
def randf(a,b): return random.uniform(a,b)

def hard_freeze(paint_white=True):
    """crash!!"""
    try:
        screen = pygame.display.get_surface()
        if paint_white and screen:
            wh = pygame.Surface(screen.get_size())
            wh.fill((255,255,255))
            screen.blit(wh, (0,0))
            pygame.display.flip()
    except Exception:
        pass
    try:
        pygame.event.set_blocked(None)  # block all events
    except Exception:
        pass
    
    while True:
        time.sleep(0.1)

class Camera:
    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.pos = [0.0, 0.0, 0.0]
        self.speed = 1.1

    def forward_flat(self):
        return (math.sin(self.yaw), 0.0, math.cos(self.yaw))
    def right(self):
        return (math.cos(self.yaw), 0.0, -math.sin(self.yaw))

    def apply(self, v):
        x,y,z = v[0]-self.pos[0], v[1]-self.pos[1], v[2]-self.pos[2]
        cy, sy = math.cos(-self.yaw), math.sin(-self.yaw)
        x, z = cy*x - sy*z, sy*x + cy*z
        cp, sp = math.cos(-self.pitch), math.sin(-self.pitch)
        y, z = cp*y - sp*z, sp*y + cp*z
        return (x, y, z)

def project(v):
    x,y,z = v
    z = (z + 3.0)
    if z <= 0.1: z = 0.1
    f = 780 / z
    return (WIDTH/2 + x*f, HEIGHT/2 - y*f)

def draw_cube(screen, cam, pos=(0,0,0), size=0.35, col=(180, 20, 40)):
    s = size*0.5
    verts = [(-s,-s,-s),( s,-s,-s),( s, s,-s),(-s, s,-s),
             (-s,-s, s),( s,-s, s),( s, s, s),(-s, s, s)]
    verts = [(vx+pos[0], vy+pos[1], vz+pos[2]) for (vx,vy,vz) in verts]
    faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]
    depths = []
    for i, fidx in enumerate(faces):
        zavg = sum(cam.apply(verts[idx])[2] for idx in fidx)/4.0
        depths.append((zavg, i))
    depths.sort(reverse=True)
    for _, i in depths:
        pts = [project(cam.apply(verts[idx])) for idx in faces[i]]
        pygame.draw.polygon(screen, col, pts, 0)

def draw_glitchy_mass(screen, cam, t, pos=(0,0,0), base_radius=0.25, layers=5):
    random.seed(int(t*10))
    for k in range(layers):
        n = random.randint(5, 8)
        pts3 = []
        radius = base_radius * (1.0 + 0.12*k)
        for i in range(n):
            theta = randf(0, 2*math.pi)
            phi = randf(-0.8, 0.8)
            rj = radius * (0.85 + randf(-0.18, 0.18)) * (1.0 + 0.05*math.sin(t*3 + i+k))
            x = math.cos(theta) * math.cos(phi) * rj
            y = math.sin(phi) * rj
            z = math.sin(theta) * math.cos(phi) * rj
            x += 0.05*math.sin((i+k)*1.7 + t*4.0)
            y += 0.05*math.cos((i-k)*1.3 + t*3.3)
            z += 0.05*math.sin((i+k*2)*1.1 - t*2.7)
            pts3.append((x+pos[0], y+pos[1], z+pos[2]))
        pts2 = [project(cam.apply(p)) for p in pts3]
        g = clamp(150 + random.randint(-30, 30) - k*12, 60, 200)
        pygame.draw.polygon(screen, (g,g,g), pts2, 0)

class Hunter:
    def __init__(self, start_pos, speed=0.35):
        self.pos = list(start_pos)
        self.speed = speed
        self.seed = random.random()*1000.0
        self.size = 0.35

    def update(self, cam, dt):
        dx = cam.pos[0] - self.pos[0]
        dy = cam.pos[1] - self.pos[1]
        dz = cam.pos[2] - self.pos[2]
        L = (dx*dx + dy*dy + dz*dz) ** 0.5 or 1.0
        step = self.speed * dt
        self.pos[0] += (dx/L) * step
        self.pos[1] += (dy/L) * step
        self.pos[2] += (dz/L) * step
        if L < 0.4:
            hard_freeze(paint_white=True)  # crash on contact

    def draw(self, screen, cam, t):
        random.seed(int(self.seed + t*20))
        n = random.randint(6, 9)
        radius = self.size * (0.9 + 0.25*math.sin(t*2.2 + self.seed))
        pts3 = []
        for i in range(n):
            ang = randf(0, 2*math.pi)
            elev = randf(-0.9, 0.9)
            r = radius * (0.8 + randf(-0.2, 0.2))
            x = math.cos(ang)*math.cos(elev)*r + self.pos[0]
            y = math.sin(elev)*r + self.pos[1]
            z = math.sin(ang)*math.cos(elev)*r + self.pos[2]
            x += 0.12*math.sin(ang*3 + t*7)
            y += 0.12*math.cos(ang*2 + t*6)
            z += 0.12*math.sin(ang*4 - t*5)
            pts3.append((x,y,z))
        pts2 = [project(cam.apply(p)) for p in pts3]
        col = (60, 10, 20)
        pygame.draw.polygon(screen, col, pts2, 0)

def main():
    pygame.init()
    pygame.display.set_caption("redboxxx")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    cam = Camera()
    mouse_locked = False
    invert_pitch = False

    hunters = []

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        t = pygame.time.get_ticks() / 1000.0
        wheel_move = 0.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
              
                hard_freeze(paint_white=True)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                
                    hard_freeze(paint_white=True)
                elif e.key == pygame.K_l:
                    mouse_locked = not mouse_locked
                    pygame.mouse.set_visible(not mouse_locked)
                    pygame.event.set_grab(mouse_locked)
                elif e.key == pygame.K_i:
                    invert_pitch = not invert_pitch
                elif e.key == pygame.K_m:
                    fx, _, fz = cam.forward_flat()
                    spawn = (cam.pos[0] + fx*14.0, cam.pos[1], cam.pos[2] + fz*14.0)
                    hunters.append(Hunter(spawn, speed=0.55))
                elif e.key == pygame.K_z:
                    cam.pos = [0.0,0.0,0.0]; cam.yaw = 0.0; cam.pitch = 0.0
                    hunters.clear()
            elif e.type == pygame.MOUSEWHEEL:
                wheel_move += e.y

        if mouse_locked:
            mx, my = pygame.mouse.get_rel()
            cam.yaw += mx * 0.0032
            cam.pitch = clamp(cam.pitch + (my * 0.0032 if invert_pitch else -my * 0.0032), -1.2, 1.2)
        else:
            pygame.mouse.get_rel()

        keys = pygame.key.get_pressed()
        boost = 2.5 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0
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
            f = 780.0
            rx_cam = (mx - WIDTH/2) / f
            ry_cam = -(my - HEIGHT/2) / f
            rz_cam = 1.0
            cy, sy = math.cos(cam.yaw), math.sin(cam.yaw)
            cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
            x1, y1, z1 = rx_cam, ry_cam*cp + rz_cam*sp, -ry_cam*sp + rz_cam*cp
            dx, dy, dz = x1*cy - z1*sy, y1, x1*sy + z1*cy
            L = (dx*dx + dy*dy + dz*dz) ** 0.5 or 1.0
            dx, dy, dz = dx/L, dy/L, dz/L
            wheel_speed = 1.6 * wheel_move * (1.0 + 0.5*boost)
            cam.pos[0] += dx * wheel_speed * dt * 60.0
            cam.pos[1] += dy * wheel_speed * dt * 60.0
            cam.pos[2] += dz * wheel_speed * dt * 60.0

        for h in list(hunters):
            h.update(cam, dt)

        # Render
        screen.fill((96, 10, 22)) 
        draw_cube(screen, cam, pos=(0,0,0), size=0.35, col=(180, 20, 40))
        draw_glitchy_mass(screen, cam, t, pos=(0,0,0), base_radius=0.28, layers=6)
        for h in hunters:
            h.draw(screen, cam, t)

        font = pygame.font.SysFont("consolas,monospace", 16)
        hud = f"[Hunters {len(hunters)}] [FPS {clock.get_fps():.1f}]"
        help1 = "L lock | Wheel move-to-cursor | WASD+QE | Shift | M spawn crasher (touch = crash) | ESC = FREEZE | I invert | Z reset"
        screen.blit(font.render(hud, True, (245,230,235)), (10, 8))
        screen.blit(font.render(help1, True, (210,180,190)), (10, 28))

        pygame.display.flip()

  
    pygame.quit()

if __name__ == '__main__':
    main()
